import json
import os
import pickle
import time
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    BertModel, BertPreTrainedModel,
    AdamW, get_linear_schedule_with_warmup,
    TrainingArguments,
)

from datasets import Dataset, concatenate_datasets, load_from_disk, load_dataset, Features, DatasetDict, Sequence, Value
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm


# Dense Retrieval
class DenseRetrieval:

    def __init__(
        self, args, num_neg, tokenizer, p_encoder, q_encoder,
        data_path: Optional[str] = "../data",
        train_context_path: Optional[str] = "train_dataset",
        test_context_path: Optional[str] = "wikipedia_documents.json"
        ):

        '''
        학습과 추론에 사용될 여러 셋업을 마쳐봅시다.
        '''
        self.data_path = data_path
        dataset = load_from_disk(os.path.join(data_path, train_context_path))
        dataset = self.concat_data(dataset)
        train_dataset = dataset['train']
        # valid_dataset = dataset['validation']
        
        # num_sample = 20
        # sample_idx = np.random.choice(range(len(train_dataset)), num_sample)
        # train_dataset = train_dataset[sample_idx]
        self.dataset = train_dataset

        self.args = args
        self.num_neg = num_neg

        with open(os.path.join(data_path, test_context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        # for debugging
        # temp_wiki = {}
        # for key in wiki.keys():
        #     if key == '100':
        #         break
        #     else:
        #         temp_wiki[key] = wiki[key]
        # wiki = temp_wiki
        
        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로        
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder

        # Transform by vectorizer
        self.tfidfv = TfidfVectorizer(
            tokenizer=tokenizer.tokenize, ngram_range=(1, 2), max_features=50000,
        )

        # self.get_sparse_embedding()
        self.prepare_in_batch_negative(num_neg=num_neg)

    def concat_data(self, dataset):
        korquad_dataset = load_dataset("squad_kor_v1")['train']
        
        train = pd.DataFrame(dataset['train'])
        val = pd.DataFrame(dataset['validation'])
        korquad = pd.DataFrame(korquad_dataset)
        
        datasets = pd.concat([train, val])
        datasets = datasets[['answers','context','id','question']]
        korquad = korquad[['answers','context','id','question']]
        datasets = pd.concat([datasets,korquad],ignore_index=True)
        print('#'*100)
        print(datasets)

        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "answer_start": Value(dtype="int32", id=None),
                        "text": Value(dtype="string", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

        datasets = DatasetDict({"train": Dataset.from_pandas(datasets, features=f)})
        return datasets

    
    def get_sparse_embedding(self) -> NoReturn:

        """
        Summary:
            Passage Embedding을 만들고
            TFIDF와 Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        # Pickle을 저장합니다.
        pickle_name = f"sparse_embedding_train.bin"
        tfidfv_name = f"tfidv_train.bin"
        emd_path = os.path.join(self.data_path, pickle_name)
        tfidfv_path = os.path.join(self.data_path, tfidfv_name)

        if os.path.isfile(emd_path) and os.path.isfile(tfidfv_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            with open(tfidfv_path, "rb") as file:
                self.tfidfv = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            self.p_embedding = self.tfidfv.fit_transform(list(set([example for example in self.dataset['context']])))
            print(self.p_embedding.shape)
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            with open(tfidfv_path, "wb") as file:
                pickle.dump(self.tfidfv, file)
            print("Embedding pickle saved.")

    def get_relevant_doc_tfidf(self, query: str, k: Optional[int] = 20) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        # with timer("transform"):
        query_vec = self.tfidfv.transform([query])
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

    def prepare_in_batch_negative(self, dataset=None, num_neg=10, tokenizer=None):
        
        if dataset is None:
            dataset = self.dataset

        if tokenizer is None:
            tokenizer = self.tokenizer

        # 1. In-Batch-Negative 만들기
        # CORPUS를 np.array로 변환해줍니다.        
        corpus = np.array(list(set([example for example in dataset['context']])))
        p_with_neg = []

        for c in tqdm(dataset['context'], desc='negative samples: '):
            
            while True:
                neg_idxs = np.random.randint(len(corpus), size=num_neg)

                if not c in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]

                    p_with_neg.append(c)
                    p_with_neg.extend(p_neg)
                    break


        ## -- TFIDF
        # corpus = np.array(list(set([example for example in dataset['context']])))
        # p_with_neg = []

        # for q, c in zip(dataset['question'], dataset['context']):
        #     _, doc_indices = self.get_relevant_doc_tfidf(q)
        #     p_with_neg.append(c)
        #     p_neg = []
        #     for doc in corpus[doc_indices]:
        #         if doc == c:
        #             continue
        #         else:
        #             p_neg.append(doc) 
        #     p_with_neg.extend(p_neg)

        # 2. (Question, Passage) 데이터셋 만들어주기
        q_seqs = tokenizer(dataset['question'], padding="max_length", truncation=True, return_tensors='pt')
        p_seqs = tokenizer(p_with_neg, padding="max_length", truncation=True, return_tensors='pt')

        max_len = p_seqs['input_ids'].size(-1)
        p_seqs['input_ids'] = p_seqs['input_ids'].view(-1, num_neg+1, max_len)
        p_seqs['attention_mask'] = p_seqs['attention_mask'].view(-1, num_neg+1, max_len)
        p_seqs['token_type_ids'] = p_seqs['token_type_ids'].view(-1, num_neg+1, max_len)

        train_dataset = TensorDataset(
            p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'], 
            q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids']
        )

        self.train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.args.per_device_train_batch_size)

        valid_seqs = tokenizer(self.contexts, padding="max_length", truncation=True, return_tensors='pt')
        passage_dataset = TensorDataset(
            valid_seqs['input_ids'], valid_seqs['attention_mask'], valid_seqs['token_type_ids']
        )
        self.passage_dataloader = DataLoader(passage_dataset, batch_size=self.args.per_device_train_batch_size, drop_last=True)

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.
        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]
        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_indices = self.get_relevant_doc(
                    query_or_dataset["question"], k=topk
                )
                doc_indices = doc_indices.tolist()
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Dense retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def train(self, args=None):

        if args is None:
            args = self.args
        batch_size = args.per_device_train_batch_size

        # Optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        t_total = len(self.train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

        # Start training!
        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")
        f = open("train_loss.txt", "w")

        # for _ in range(int(args.num_train_epochs)):
        for _ in train_iterator:

            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:

                    self.p_encoder.train()
                    self.q_encoder.train()
            
                    targets = torch.zeros(batch_size).long() # positive example은 전부 첫 번째에 위치하므로
                    targets = targets.to(args.device)

                    p_inputs = {
                        'input_ids': batch[0].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        'attention_mask': batch[1].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        'token_type_ids': batch[2].view(batch_size * (self.num_neg + 1), -1).to(args.device)
                    }
            
                    q_inputs = {
                        'input_ids': batch[3].to(args.device),
                        'attention_mask': batch[4].to(args.device),
                        'token_type_ids': batch[5].to(args.device)
                    }
            
                    p_outputs = self.p_encoder(**p_inputs)  # (batch_size*(num_neg+1), emb_dim)
                    q_outputs = self.q_encoder(**q_inputs)  # (batch_size*, emb_dim)

                    # Calculate similarity score & loss
                    p_outputs = p_outputs.view(batch_size, self.num_neg + 1, -1)
                    q_outputs = q_outputs.view(batch_size, 1, -1)
                    print(1)
                    print(q_outputs.shape)
                    sim_scores = torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()  #(batch_size, num_neg + 1)
                    print(2)
                    print(sim_scores.shape)
                    sim_scores = sim_scores.view(batch_size, -1)
                    print(3)
                    print(sim_scores.shape)
                    sim_scores = F.log_softmax(sim_scores, dim=1)
                    print(4)
                    print(sim_scores.shape)

                    loss = F.nll_loss(sim_scores, targets)
                    f.write(str(loss)+'\n')

                    tepoch.set_postfix(loss=f'{str(loss.item())}')

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs
            torch.save(self.q_encoder.state_dict(), '/opt/ml/input/custom_code/q_encoder.pt')
            torch.save(self.p_encoder.state_dict(), '/opt/ml/input/custom_code/p_encoder.pt')
        
        f.close()


    def get_relevant_doc(self, query, k=1, args=None, p_encoder=None, q_encoder=None, tokenizer=None):
        if tokenizer is None:
            tokenizer = self.tokenizer

        if args is None:
            args = self.args

        if p_encoder is None:
            p_encoder = self.p_encoder

        if q_encoder is None:
            q_encoder = self.q_encoder

        with torch.no_grad():
            p_encoder.eval()
            q_encoder.eval()
            
            q_embs = []
            q_seqs_val = tokenizer(query, padding="max_length", truncation=True, return_tensors='pt').to(args.device)
            q_emb = q_encoder(**q_seqs_val).to('cpu')  # (num_query=1, emb_dim) tensor
            # for i in range(len(query)):
            #     ## 수정
            #     q_seqs_val = self.tokenizer(query[i], padding="max_length", truncation=True, return_tensors='pt').to(args.device)
            #     q_emb = q_encoder(**q_seqs_val).to('cpu')  # (num_query=1, emb_dim) tensor
            #     q_embs.append(q_emb.tolist())

            p_embs = []
            for batch in tqdm(self.passage_dataloader, desc='passage_dataloader: '):

                batch = tuple(t.to(args.device) for t in batch)
                p_inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2]
                }
                p_emb = p_encoder(**p_inputs).to('cpu')
                p_embs.append(p_emb)

        p_embs = torch.stack(p_embs, dim=0).view(len(p_embs)*self.args.per_device_train_batch_size, -1)  # (num_passage, emb_dim)
        
        # q_embs = torch.tensor(q_embs)
        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
        # sorted_result = np.argsort(dot_prod_scores.squeeze())[::-1]
        # doc_score = dot_prod_scores.squeeze()[sorted_result].tolist()[:k]
        # doc_indices = sorted_result.tolist()[:k]
        # return doc_score, doc_indices
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
        return rank[:, :k]
    
    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vec = self.tfidfv.transform(queries)
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices