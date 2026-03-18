import re
import json
import traceback
import ollama
import faiss
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForSequenceClassification
from transformers.generation.utils import GenerationConfig
import numpy as np
import torch
import os

import random
from sentence_transformers import SentenceTransformer
from datetime import datetime
import backoff
import logging
import argparse
import yaml
from metric import F1_scorer

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
logger = logging.getLogger()
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
choices = [
    "glm-4", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0125","chatGLM3-6b-32k", "chatGLM3-6b-8k","LongAlign-7B-64k",
    "qwen1.5-7b-chat-32k", "vicuna-v1.5-7b-16k","Llama3-8B-Instruct-8k", "Llama3-70b-8k", "Llama2-13b-chat-longlora",
    "LongRAG-chatglm3-32k", "LongRAG-qwen1.5-32k","LongRAG-vicuna-v1.5-16k", "LongRAG-llama3-8k",  "LongRAG-llama2-4k", 'llama3-8b-instruct-8k'
]

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=["hotpotqa", "2wikimultihopqa", "musique"], default="hotpotqa", help="Name of the dataset")
parser.add_argument('--top_k1', type=int, default=10, help="Number of candidates after initial retrieval")
parser.add_argument('--top_k2', type=int, default=3, help="Number of candidates after reranking")
parser.add_argument('--model', type=str, choices=choices, default="llama3-8b-instruct-8k", help="Model for generation")
parser.add_argument('--lrag_model', type=str, choices=choices, default="", help="Model for LongRAG")
parser.add_argument('--rb', action="store_true", default=False, help="Vanilla RAG")
parser.add_argument('--raw_pred', action="store_true", default=False, help="LLM direct answer without retrieval")
parser.add_argument('--rl', action="store_true", default=False, help="RAG-Long")
parser.add_argument('--ext', action="store_true", default=False, help="Only using Extractor")
parser.add_argument('--fil', action="store_true", default=False, help="Only using Extractor")
parser.add_argument('--ext_fil', action="store_true", default=True, help="Using Extractor and Filter")
parser.add_argument('--MaxClients', type=int, default=1)
parser.add_argument('--log_path', type=str, default="")
parser.add_argument('--r_path', type=str, default="../data/corpus/processed", help="Path to the vector database")
parser.add_argument('--input_json', type=str, default="../data/hotpotqa/hotpot_dev_distractor_v1_dev_200.json", help="Input dataset json/jsonl (optional, for prediction only)")
parser.add_argument('--output_json', type=str, default="../output/hotpotqa-dev-200.json", help="Output predictions json path")
args = parser.parse_args()



def get_word_len(input):
    tokenized_prompt = set_prompt_tokenizer(input, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
    return len(tokenized_prompt)

def set_prompt(input, maxlen):
    tokenized_prompt = set_prompt_tokenizer(input, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if len(tokenized_prompt) > maxlen:
        half = int(maxlen * 0.5)
        input = set_prompt_tokenizer.decode(
            tokenized_prompt[:half], skip_special_tokens=True
        ) + set_prompt_tokenizer.decode(
            tokenized_prompt[-half:], skip_special_tokens=True
        )
    return input, len(tokenized_prompt)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(model2path, model_name):
   
    if "gpt" in model_name or "glm-4" in model_name or "glm3-turbo-128k" in model_name:
        return model_name, model_name

    path = model2path.get(model_name, "")

    
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name or "longalign-6b" in model_name or "qwen" in model_name or "llama3" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
    elif "llama2" in model_name:
        tokenizer = LlamaTokenizer.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map='auto')

    model = model.eval()
    return model, tokenizer

def ollama_embed_texts(texts, model_name, batch_size=64):
    """
    
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = ollama.embed(model=model_name, input=batch)   # resp["embeddings"] -> List[List[float]]
        vecs = np.array(resp["embeddings"], dtype=np.float32)

        # normalize for cosine similarity with IndexFlatIP
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        vecs = vecs / norms

        all_vecs.append(vecs)

    return np.vstack(all_vecs).astype(np.float32)


@backoff.on_exception(backoff.expo, (Exception), max_time=200)
def pred(model_name, model, tokenizer, prompt, maxlen, max_new_tokens=32, temperature=1):
    try:

        if "longalign" in model_name.lower() and max_new_tokens == 32:
            max_new_tokens = 128
        prompt, prompt_len = set_prompt(prompt, maxlen)

        if isinstance(mpath, str) and ":" in mpath and not os.path.exists(mpath):
            # NOTE: pass the real ollama model name like "llama3:8b"
            resp = ollama.chat(
                model=mpath,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": max_new_tokens, "temperature": temperature}
            )
            response = resp.get("message", {}).get("content", "")
            if not response:
                raise RuntimeError(f"Ollama model '{mpath}' returned empty response")
            return response, prompt_len

        history = []

        # 3) ChatGLM / InternLM / LongAlign (use .chat)
        if "internlm" in model_name or "chatglm" in model_name or "longalign-6b" in model_name:
            response, history = model.chat(
                tokenizer, prompt, history=history,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_beams=1, do_sample=False
            )
            return response, prompt_len

        # 4) Baichuan (if enabled)
        if "baichuan" in model_name:
            messages = [{"content": prompt, "role": "user"}]
            model.generation_config = GenerationConfig.from_pretrained(
                model2path["baichuan2-7b-4k"],
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                num_beams=1, do_sample=False
            )
            response = model.chat(tokenizer, messages)
            return response, prompt_len

        # 5) HF Llama3 (only when model2path is a local path or HF repo)
        if "llama3" in model_name:
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)

            terminators = [tokenizer.eos_token_id]
            eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if isinstance(eot_id, int) and eot_id >= 0:
                terminators.append(eot_id)

            outputs = model.generate(
                input_ids,
                eos_token_id=terminators,
                temperature=temperature,
                num_beams=1, do_sample=False
            )
            response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
            return response, prompt_len

        # 8) Other llama models (e.g., llama2 use [INST])
        if "llama" in model_name:
            inputs = tokenizer(f"[INST]{prompt}[/INST]", truncation=False, return_tensors="pt").to(model.device)
        else:
            inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to(model.device)

        context_length = inputs.input_ids.shape[-1]
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1, do_sample=False,
            temperature=temperature
        )
        response = tokenizer.decode(output[0][context_length:], skip_special_tokens=True).strip()
        return response, prompt_len

    except Exception as e:
        print(f"[pred ERROR] model={model_name} err={e}")
        traceback.print_exc()
        time.sleep(1)
        return None, None

def setup_logger(logger, filename='log'):
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter(fmt="[%(asctime)s][%(levelname)s] - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(os.path.join(log_path, filename))
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)

def print_args(args):
    logger.info(f"{'*' * 30} CONFIGURATION {'*' * 30}")
    for key, val in sorted(vars(args).items()):
        keystr = f"{key}{' ' * (30 - len(key))}"
        logger.info(f"{keystr} --> {val}")
    logger.info(f"LongRAG model used: {args.lrag_model}")
    logger.info(f"{'*' * 30} CONFIGURATION {'*' * 30}")

def search_q(question):
    doc_len = {}
    raw_pred = ""
    if args.raw_pred:
        raw_pred = search_cache_and_predict(raw_pred, f'{log_path}/raw_pred.json', 'raw_pred', question, model_name, model, tokenizer, lambda: create_prompt(question), maxlen)

    retriever, match_id = vector_search(question)
    rerank, match_id = sort_section(question, retriever, match_id)

    fil_pred = ext_pred = ext_fil_pred = rb_pred = rl_pred = ''

    if args.ext_fil:
        ext_fil_pred = load_cache(f'{log_path}/ext_fil_pred.json', 'ext_fil_pred', question, doc_len, 'E&F')
        if not ext_fil_pred:
            # dual-query merge disabled for stability at k2=3
            # plan_tmp = build_hop_plan(question, rerank)
            # rewrite_q = _rewrite_query(question, plan_tmp)
            # retr2, mid2 = vector_search(rewrite_q)
            # merged = retriever + retr2
            # merged_ids = match_id + mid2
            # rerank, match_id = sort_section(question, merged, merged_ids)

            plan = build_hop_plan(question, rerank)
            packs = build_evidence_packs(rerank, match_id, plan, question)
            terms = _plan_terms(plan)
            # k2=3: do NOT drop evidence further; keep all packs (<=3) even if from same doc
            # packs = _select_unique_packs(packs, match_id, terms, target=3)
            combined = build_final_input(packs, plan)
            ext_fil_pred = search_cache_and_predict(
                ext_fil_pred,
                f'{log_path}/ext_fil_pred.json',
                'ext_fil_pred',
                question, model_name, model, tokenizer,
                lambda: create_prompt(combined, question),
                maxlen, doc_len, 'E&F'
            )
            # adaptive re-retrieval if answer is low confidence
            if _is_bad_answer(ext_fil_pred):
                rewrite_q = _rewrite_query(question, plan)
                retr2, mid2 = vector_search(rewrite_q)
                rer2, mid2 = sort_section(question, retr2, mid2)
                packs2 = build_evidence_packs(rer2, mid2, plan, question)
                # packs2 = _select_unique_packs(packs2, mid2, terms, target=3)
                merged = packs + [p for p in packs2 if p not in packs]
                # merged = merged[:3]
                combined2 = build_final_input(merged, plan)
                cand = search_cache_and_predict(
                    ext_fil_pred,
                    f'{log_path}/ext_fil_pred.json',
                    'ext_fil_pred',
                    question, model_name, model, tokenizer,
                    lambda: create_prompt(combined2, question),
                    maxlen, doc_len, 'E&F', force=True
                )
                cand0 = cand.strip() if cand else ""
                if looks_like_short_answer(cand0) and (cand0 in combined2):
                    ext_fil_pred = cand0

    if args.rb:
        rb_pred = load_cache(f'{log_path}/rb_pred.json', 'rb_pred', question, doc_len, 'R&B')
        if not rb_pred:
            rb_pred = search_cache_and_predict(rb_pred, f'{log_path}/rb_pred.json', 'rb_pred', question, model_name, model, tokenizer, lambda: create_prompt(''.join(rerank), question), maxlen, doc_len, 'R&B')
    
    if args.rl:
        rl_pred = load_cache(f'{log_path}/rl_pred.json', 'rl_pred', question, doc_len, 'R&L')
        if not rl_pred:
            rl_pred = search_cache_and_predict(rl_pred, f'{log_path}/rl_pred.json', 'rl_pred', question, model_name, model, tokenizer, lambda: create_prompt(''.join(s2l_doc(rerank, match_id, maxlen)[0]), question), maxlen, doc_len, 'R&L')
    
    return question, retriever, rerank, raw_pred, rb_pred, ext_pred, fil_pred, rl_pred, ext_fil_pred, doc_len

def load_cache(cache_path, pred_key, question, doc_len=None, doc_key=None):
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                data = json.loads(line)
                if data['question'] == question:
                    pred_result = data[pred_key]
                    if doc_len is not None and doc_key is not None:
                        doc_len[doc_key] = data["input_len"]
                    return pred_result
    return ''

def search_cache_and_predict(pred_result, cache_path, pred_key, question, model_name, model, tokenizer, create_prompt_func, maxlen, doc_len=None, doc_key=None, force=False):
    if force or (not pred_result):
        query = create_prompt_func()
        pred_result, input_len = pred(model_name, model, tokenizer, query, maxlen)
        with open(cache_path, 'a', encoding='utf-8') as f:
            json.dump({'question': question, pred_key: pred_result, "input_len": input_len}, f, ensure_ascii=False)
            f.write('\n')
        if doc_len is not None and doc_key is not None:
            doc_len[doc_key] = input_len
    return pred_result

def s2l_doc(rerank, match_id, maxlen):
    unique_raw_id = []
    contents = []
    s2l_index = {}
    section_index = [id_to_rawid[str(i)] for i in match_id]
    for index, id in enumerate(section_index):
        data = raw_data[id]
        text = data["context"]
        if id in unique_raw_id and get_word_len(text) < maxlen:
            continue
        if get_word_len(text) >= maxlen:
            content = rerank[index]
        else:
            unique_raw_id.append(id)
            content = text
        s2l_index[len(contents)] = [i for i, v in enumerate(section_index) if v == section_index[index]]
        contents.append(content)
    return contents, s2l_index


def filter(question,rank_docs): 
    
    content="\n".join(rank_docs)
    query=f"{content}\n\nPlease combine the above information and give your thinking process for the following question:{question}."
    think_pro,_=pred(lrag_model_name, lrag_model, lrag_tokenizer, query,lrag_maxlen,1000)
    selected = []

    prompts=[f"""Given an article:{d}\nQuestion: {question}.\nThought process:{think_pro}.\nYour task is to use the thought process provided to decide whether you need to cite the article to answer this question. If you need to cite the article, set the status value to True. If not, set the status value to False. Please output the response in the following json format: {{"status": "{{the value of status}}"}}""" for d in rank_docs]
    pool = ThreadPool(processes=args.MaxClients)
    all_responses=pool.starmap(pred, [(lrag_model_name,lrag_model, lrag_tokenizer,prompt,lrag_maxlen,32) for prompt in prompts])

    for i,r in enumerate(all_responses):
        try:    
            result=json.loads(r[0])
            res=result["status"] 
            if len(all_responses)!=len(rank_docs):
                break     
            if res.lower()=="true":
                selected.append(rank_docs[i])
        except:
            match=re.search("True|true",r[0])
            if match:
                selected.append(rank_docs[i])
    if len(selected)==0:
        selected=rank_docs
    return selected


def r2long_unique(rerank, match_id):
    unique_raw_id = list(set(id_to_rawid[str(i)] for i in match_id))
    section_index = [id_to_rawid[str(i)] for i in match_id]
    contents = [''.join(rerank[i] for i in range(len(section_index)) if section_index[i] == uid) for uid in unique_raw_id]
    return contents, unique_raw_id

def extractor(question, docs, match_id):
    long_docs = s2l_doc(docs, match_id, lrag_maxlen)[0]
    content = ''.join(long_docs)
    query = f"{content}.\n\nBased on the above background, please output the information you need to cite to answer the question below.\n{question}"
    response = pred(lrag_model_name, lrag_model, lrag_tokenizer, query, lrag_maxlen, 1000)[0]
    # logger.info(f"cite_passage responses: {all_responses}")
    return [response]


def vector_search(question):
    feature = ollama_embed_texts([question], emb_model_name)  # (1, D) np.float32
    distance, match_id = vector.search(feature, args.top_k1)
    content = [chunk_data[int(i)] for i in match_id[0]]
    return content, list(match_id[0])


def sort_section(question, section, match_id):
    if not section:
        return [], []
    if len(match_id) != len(section):
        n = min(len(match_id), len(section))
        section = section[:n]
        match_id = match_id[:n]
    q = [question] * len(section)
    features = cross_tokenizer(q, section, padding=True, truncation=True, return_tensors="pt").to(device)
    cross_model.eval()
    with torch.no_grad():
        scores = cross_model(**features).logits.squeeze(dim=1)
    result = [section[sort_scores[i].item()] for i in range(k)]
    match_id = [match_id[sort_scores[i].item()] for i in range(k)]
    return result, match_id

def create_prompt(input, question):
    user_prompt = (
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        f"The following are given passages.\n{input}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        f"Question: {question}\nAnswer:"
    )
    return user_prompt

def extract_question_terms(question: str):
    terms = set()
    for m in re.findall(r'"([^"]+)"', question):
        if 2 <= len(m) <= 40:
            terms.add(m.strip())
    for m in re.findall(r'\b\d{2,4}\b', question):
        terms.add(m)
    for m in re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b', question):
        terms.add(m.strip())
    words = [w.strip(".,!?;:()[]{}").lower() for w in question.split()]
    words = [w for w in words if len(w) >= 5]
    for w in words[:6]:
        terms.add(w)
    return list(terms)

def split_into_sentences(text):
    if not text:
        return []
    stop_list = ['  ', '  ', '?', '  ', '!', '  ', ';']
    split_pattern = f"({'|'.join(map(re.escape, stop_list))})"
    parts = re.split(split_pattern, text)
    if len(parts) == 1:
        return [text]
    sentences = [parts[i] + parts[i + 1] for i in range(0, len(parts) - 1, 2)]
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1])
    return [s.strip() for s in sentences if s.strip()]

def build_hop_plan(question, rerank_docs):
    plan_prompt = (
        "You are a retrieval planner. Return a compact JSON only.\n"
        "JSON schema:\n"
        "{\n"
        "  \"hops\": [\"step1\", \"step2\", \"step3\"],\n"
        "  \"key_entities\": [\"e1\", \"e2\", \"e3\"],\n"
        "  \"keywords\": {\"step1\": [\"k1\", \"k2\"], \"step2\": [\"k3\"], \"step3\": [\"k4\"]}\n"
        "}\n"
        "Constraints: at most 3 hops, 8 entities, each step 3-6 keywords.\n"
        "If you cannot infer hops, still return 1 hop with keywords directly from the question.\n"
        f"Question: {question}\n"
        f"Anchors:\n" + "\n\n---\n\n".join(rerank_docs) + "\n"
        "Return JSON only."
    )
    raw, _ = pred(model_name, model, tokenizer, plan_prompt, maxlen, max_new_tokens=256)
    try:
        plan = json.loads(raw)
    except Exception:
        plan = {"hops": [], "key_entities": [], "keywords": {}}
    # normalize
    hops = plan.get("hops", [])[:3] if isinstance(plan.get("hops", []), list) else []
    entities = plan.get("key_entities", [])[:8] if isinstance(plan.get("key_entities", []), list) else []
    keywords = plan.get("keywords", {}) if isinstance(plan.get("keywords", {}), dict) else {}
    return {"hops": hops, "key_entities": entities, "keywords": keywords}

def _plan_terms(plan):
    terms = []
    for e in plan.get("key_entities", []):
        if isinstance(e, str) and e.strip():
            terms.append(e.strip())
    kw = plan.get("keywords", {})
    if isinstance(kw, dict):
        for _, lst in kw.items():
            if isinstance(lst, list):
                for k in lst:
                    if isinstance(k, str) and k.strip():
                        terms.append(k.strip())
    return list(dict.fromkeys(terms))

def _relation_terms():
    return [
        "seat of", "capital", "located in", "born in", "founded", "opened", "won",
        "member of", "part of", "headquartered", "directed by", "composed by",
        "author of", "written by", "starring", "released", "premiered"
    ]

def _sentence_score(sentence, question, terms):
    s = sentence.lower()
    score = 0
    # terms from plan/question
    for t in terms:
        if t.lower() in s:
            score += 2
    # relation cues
    for r in _relation_terms():
        if r in s:
            score += 2
    # token overlap
    q_tokens = [w.strip(".,!?;:()[]{}").lower() for w in question.split()]
    score += sum(1 for w in q_tokens if w and w in s)
    return score

def build_evidence_packs(rerank_docs, match_id, plan, question, window=450, max_sents=10):
    packs = []
    plan_terms = _plan_terms(plan)
    q_terms = extract_question_terms(question)
    terms = list(dict.fromkeys(plan_terms + q_terms))
    for i, chunk in enumerate(rerank_docs):
        rawid = id_to_rawid.get(str(match_id[i]))
        try:
            rawid_int = int(rawid)
        except Exception:
            rawid_int = None
        doc_text = raw_data[rawid_int].get("context", "") if rawid_int is not None and 0 <= rawid_int < len(raw_data) else ""
        if not doc_text:
            packs.append(chunk)
            continue
        pos = doc_text.find(chunk)
        if pos == -1 and terms:
            # fallback: first occurrence of any term
            for t in terms:
                p = doc_text.lower().find(t.lower())
                if p != -1:
                    pos = p
                    break
        if pos == -1:
            window_text = doc_text[:1200]
        else:
            start = max(0, pos - window)
            end = min(len(doc_text), pos + len(chunk) + window)
            window_text = doc_text[start:end]

        sents = split_into_sentences(window_text)
        scored = [(s, _sentence_score(s, question, terms)) for s in sents]
        scored.sort(key=lambda x: x[1], reverse=True)
        best_sent = scored[0][0] if scored else ""
        best_idx = next((j for j, s in enumerate(sents) if s == best_sent), 0)
        L = max(0, best_idx - 2)
        R = min(len(sents), best_idx + 3)
        selected = sents[L:R]
        # ensure at least one relation sentence if possible
        if sents:
            rel = None
            for s, _ in scored:
                if any(r in s.lower() for r in _relation_terms()):
                    rel = s
                    break
            if rel and rel not in selected:
                selected[-1:] = [rel]
        # dedupe and keep order
        seen = set()
        compact = []
        for s in selected:
            if s not in seen:
                compact.append(s)
                seen.add(s)
        if compact and chunk not in " ".join(compact):
            compact.insert(0, chunk)
        packs.append(" ".join(compact) if compact else chunk)
    return packs

def build_final_input(packs, plan):
    parts = []
    for i, p in enumerate(packs):
        parts.append(f"[Pack {i+1}]\n{p}")
    parts.append("[Hop Plan]\n" + json.dumps(plan, ensure_ascii=False))
    return "\n\n".join(parts)

def _postprocess_answer(ans):
    if ans is None:
        return ans
    text = ans.strip()
    # remove trailing punctuation
    text = text.rstrip(" .,:;")
    # simple numeric to word / ordinal
    num_map = {
        "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
        "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
        "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen",
        "15": "fifteen", "16": "sixteen", "17": "seventeen", "18": "eighteen",
        "19": "nineteen", "20": "twenty"
    }
    ord_map = {
        "1": "first", "2": "second", "3": "third", "4": "fourth", "5": "fifth",
        "6": "sixth", "7": "seventh", "8": "eighth", "9": "ninth", "10": "tenth",
        "11": "eleventh", "12": "twelfth", "13": "thirteenth", "14": "fourteenth",
        "15": "fifteenth", "16": "sixteenth", "17": "seventeenth", "18": "eighteenth",
        "19": "nineteenth", "20": "twentieth"
    }
    # replace pure number with word
    if text.isdigit() and text in num_map:
        text = num_map[text]
    # consecutive -> ordinal
    if "consecutive" in text.lower():
        parts = text.split()
        if parts and parts[0].isdigit() and parts[0] in ord_map:
            parts[0] = ord_map[parts[0]]
            text = " ".join(parts)
    return text

def _rewrite_query(question, plan):
    terms = _plan_terms(plan)
    ent = plan.get("key_entities", [])
    kws = []
    kw = plan.get("keywords", {})
    if isinstance(kw, dict):
        for _, lst in kw.items():
            if isinstance(lst, list):
                kws.extend([k for k in lst if isinstance(k, str)])
    ent = [e for e in ent if isinstance(e, str)]
    return " ".join([question] + ent[:5] + kws[:8])

def _is_bad_answer(ans):
    if ans is None:
        return True
    t = ans.strip().lower()
    if t == "unknown":
        return True
    if len(t) < 2 and t not in {"yes", "no"}:
        return True
    bad_phrases = [
        "not enough information", "cannot be determined", "cannot be found",
        "no information", "not provided", "i don't know", "unable to"
    ]
    if any(p in t for p in bad_phrases):
        return True
    if len(t) > 80:
        return True
    return False

def looks_like_short_answer(a: str) -> bool:
    if a is None:
        return False
    t = a.strip()
    if not t:
        return False
    tl = t.lower()
    if tl == "unknown":
        return False
    if len(t) > 60:
        return False
    if any(p in t for p in [".", "?", "!", ";", ":", "  ", "  ", "  ", "  ", "  ", "\n"]):
        return False
    return True

def _pack_score(pack, terms):
    if not pack:
        return 0
    p = pack.lower()
    return sum(1 for t in terms if t.lower() in p)

def _select_unique_packs(packs, match_ids, terms, target=3):
    by_raw = {}
    for i, p in enumerate(packs):
        rawid = id_to_rawid.get(str(match_ids[i]))
        key = rawid
        score = _pack_score(p, terms)
        if key not in by_raw or score > by_raw[key][0]:
            by_raw[key] = (score, p)
    selected = [v[1] for v in by_raw.values()]
    return selected[:target]


if __name__ == '__main__':
    seed_everything(42)
    index_path = f'{args.r_path}/{args.dataset}/vector.index' # Vector index path
    vector = faiss.read_index(index_path)
    raw_data = []
    with open(f'../data/corpus/raw/{args.dataset}.jsonl', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_data.append(json.loads(line))

    with open(f'{args.r_path}/{args.dataset}/id_to_rawid.json', encoding='utf-8') as f:
        id_to_rawid = json.load(f)
    with open(f"{args.r_path}/{args.dataset}/chunks.json", "r", encoding='utf-8') as fin:
        chunk_data = json.load(fin)

    now = datetime.now()
    now_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = args.log_path or f'./log/{args.r_path.split("/")[-1]}/{args.dataset}/{args.model}/{args.lrag_model or "base"}/{now_time}'
    os.makedirs(log_path, exist_ok=True)

    with open("../config/config.yaml", "r", encoding='utf-8') as file:
        config = yaml.safe_load(file)

    model_name = args.model.lower()
    model2path = config["model_path"]
    maxlen = config["model_maxlen"][model_name]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # emb_model = SentenceTransformer(model2path["emb_model"]).to(device)
    emb_model_name = model2path["emb_model"]
    # cross_tokenizer = AutoTokenizer.from_pretrained(model2path["rerank_model"])
    # cross_model = AutoModelForSequenceClassification.from_pretrained(model2path["rerank_model"]).to(device)
    cross_tokenizer = AutoTokenizer.from_pretrained(
        model2path["rerank_model"],
        local_files_only=True,
        trust_remote_code=True
    )
    cross_model = AutoModelForSequenceClassification.from_pretrained(
        model2path["rerank_model"],
        local_files_only=True,
        trust_remote_code=True
    ).to(device)

    model, tokenizer = load_model_and_tokenizer(model2path, model_name)

    if args.lrag_model:
        lrag_model_name = args.lrag_model.lower()
        lrag_maxlen = config["model_maxlen"][lrag_model_name]
        lrag_model, lrag_tokenizer = (model, tokenizer) if model_name == lrag_model_name else load_model_and_tokenizer(model2path, lrag_model_name)
    else:
        lrag_model_name, lrag_model, lrag_tokenizer, lrag_maxlen = (model_name, model, tokenizer, maxlen)
    set_prompt_tokenizer = AutoTokenizer.from_pretrained(r"../model/llama3-tokenizer", trust_remote_code=True, local_files_only=True)
    setup_logger(logger)
    print_args(args)

    ids, questions, answer, raw_preds, rank_preds, ext_preds, fil_preds, longdoc_preds, ext_fil_preds, docs_len = [], [], [], [], [], [], [], [], [], []
    qs_data = []
    if args.input_json:
        with open(args.input_json, encoding='utf-8') as f:
            if args.input_json.endswith('.jsonl'):
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    qs_data.append(json.loads(line))
            else:
                qs_data = json.load(f)
    else:
        with open(f'../data/eval/{args.dataset}.jsonl', encoding='utf-8') as f:
            # qs_data = json.load(f)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                qs_data.append(json.loads(line))

    for d in qs_data:
        ids.append(d.get("_id", str(len(ids))))
        questions.append(d["question"])
        answer.append(d["answer"])

    pred_answer = {}
    pred_sp = {}
    for index, query in tqdm(enumerate(questions), total=len(questions)):
        logger.info(f"Question: {query}")
        question, retriever, rerank, raw_pred, rb_pred, ext_pred, fil_pred, rl_pred, ext_fil_pred, doc_len = search_q(query)

        # raw_preds.append(raw_pred)
        # rank_preds.append(rb_pred)
        # ext_preds.append(ext_pred)
        # fil_preds.append(fil_pred)
        # longdoc_preds.append(rl_pred)
        ext_fil_preds.append(ext_fil_pred)
        docs_len.append(doc_len)

        pred_text = ""
        if args.ext_fil:
            pred_text = ext_fil_pred
        elif args.ext:
            pred_text = ext_pred
        elif args.fil:
            pred_text = fil_pred
        elif args.rl:
            pred_text = rl_pred
        elif args.rb:
            pred_text = rb_pred
        elif args.raw_pred:
            pred_text = raw_pred
        pred_answer[ids[index]] = _postprocess_answer(pred_text)
        pred_sp[ids[index]] = []

    all_len1 = all_len2 = all_len3 = all_len4 = all_len5 = 0
    for dl in docs_len:
        # all_len1 += dl.get('Ext', 0)
        # all_len2 += dl.get('Fil', 0)
        # all_len3 += dl.get('R&B', 0)
        # all_len4 += dl.get('R&L', 0)
        all_len5 += dl.get('E&F', 0)

    doc_len_eval = {
        # "Ext": all_len1 / len(docs_len),
        # "Fil": all_len2 / len(docs_len),
        # "R&B": all_len3 / len(docs_len),
        # "R&L": all_len4 / len(docs_len),
        "E&F": all_len5 / len(docs_len)
    }
    
    
    F1 = {
        # "raw_pre": F1_scorer(raw_preds, answer),
        # "R&B": F1_scorer(rank_preds, answer),
        # "Ext": F1_scorer(ext_preds, answer),
        # "Fil": F1_scorer(fil_preds, answer),
        # "R&L": F1_scorer(longdoc_preds, answer),
        "E&F": F1_scorer(ext_fil_preds, answer)
    }

    eval_result = {"F1": F1, "doc_len": doc_len_eval}
    with open(f"{log_path}/eval_result.json", "w") as fout:
        json.dump(eval_result, fout, ensure_ascii=False, indent=4)

    if args.output_json:
        out_path = args.output_json
    else:
        os.makedirs("output", exist_ok=True)
        out_path = os.path.join("output", f"{args.dataset}-k-{args.top_k1}.json")
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"answer": pred_answer, "sp": pred_sp}, f, ensure_ascii=False)
    logger.info(f"Saved predictions to {out_path}")

