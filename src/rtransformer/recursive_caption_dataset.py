import copy
import torch
import logging
import math
import nltk
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
import json
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

log_format = "%(asctime)-10s: %(message)s"
logging.basicConfig(level=logging.INFO)


class RecursiveCaptionDataset(Dataset):
    PAD_TOKEN = "[PAD]"  # padding of the whole sequence, note
    CLS_TOKEN = "[CLS]"  # leading token of the joint sequence
    SEP_TOKEN = "[SEP]"  # a separator for video and text
    VID_TOKEN = "[VID]"  # used as placeholder in the clip+text joint sequence
    BOS_TOKEN = "[BOS]"  # beginning of the sentence
    EOS_TOKEN = "[EOS]"  # ending of the sentence
    UNK_TOKEN = "[UNK]"
    PAD = 0
    CLS = 1
    SEP = 2
    VID = 3
    BOS = 4
    EOS = 5
    UNK = 6
    IGNORE = -1  # used to calculate loss

    def __init__(self, dset_name, data_dir, video_feature_dir, region_feature_dir, region_kg_dir, asr_path, asrkg_path, outkg_path, duration_file, word2idx_path,
                 max_t_len, max_v_len, max_n_sen, max_a_len, region_feature_300dir, mode="train", recurrent=True, untied=False):
        self.dset_name = dset_name
        self.word2idx = load_json(word2idx_path)
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self.data_dir = data_dir  # containing training data
        self.video_feature_dir = video_feature_dir  # a set of .h5 files
        self.region_feature_dir =  region_feature_dir
        self.region_feature_300dir = region_feature_300dir
        self.region_kg_dir = load_json(region_kg_dir)
        self.asr_path = load_json(asr_path)
        self.asrkg = load_json(asrkg_path)
        self.outkg = load_json(outkg_path)
        self.duration_file = duration_file
        self.frame_to_second = self._load_duration()
        self.max_seq_len = max_v_len + max_t_len
        self.max_v_len = max_v_len
        self.max_t_len = max_t_len  # sen
        self.max_n_sen = max_n_sen
        self.max_asr_len = max_a_len

        self.max_kg_len = 1000
        self.max_len = 492 + self.max_asr_len + self.max_t_len

        self.mode = mode
        self.recurrent = recurrent
        self.untied = untied
        assert not (self.recurrent and self.untied), "untied and recurrent cannot be True for both"

        # data entries
        self.data = None
        self.set_data_mode(mode=mode)
        self.missing_video_names = []
        self.fix_missing()

        self.num_sens = None  # number of sentence for each video, set in self._load_data()

        self.word_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(torch.load("./cache/yc2_vocab_glove_3_add7.pt")).float(),padding_idx=0)
        self.pca = PCA(n_components=300)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        items, meta = self.convert_example_to_features(self.data[index])
        return items, meta

    def set_data_mode(self, mode):
        """mode: `train` or `val`"""
        logging.info("Mode {}".format(mode))
        self.mode = mode
        if self.dset_name == "anet":
            if mode == "train":  # 10000 videos
                data_path = os.path.join(self.data_dir, "train.json")
            elif mode == "val":  # 2500 videos
                data_path = os.path.join(self.data_dir, "anet_entities_val_1.json")
            elif mode == "test":  # 2500 videos
                data_path = os.path.join(self.data_dir, "anet_entities_test_1.json")
            else:
                raise ValueError("Expecting mode to be one of [`train`, `val`, `test`], got {}".format(mode))
        elif self.dset_name == "yc2":
            if mode == "train":  # 10000 videos
                data_path = os.path.join(self.data_dir, "yc2_train_anet_format.json")
            elif mode == "val":  # 2500 videos
                data_path = os.path.join(self.data_dir, "yc2_val_anet_format.json")
            else:
                raise ValueError("Expecting mode to be one of [`train`, `val`, `test`], got {}".format(mode))
        else:
            raise ValueError
        self._load_data(data_path)

    def fix_missing(self):
        """filter our videos with no feature file"""
        for e in tqdm(self.data):
            video_name = e["name"][2:] if self.dset_name == "anet" else e["name"]
            cur_path_resnet = os.path.join(self.video_feature_dir, "{}_resnet.npy".format(video_name))
            cur_path_bn = os.path.join(self.video_feature_dir, "{}_bn.npy".format(video_name))
            for p in [cur_path_bn, cur_path_resnet]:
                if not os.path.exists(p):
                    self.missing_video_names.append(video_name)
        print("Missing {} features (clips/sentences) from {} videos".format(
            len(self.missing_video_names), len(set(self.missing_video_names))))
        # print("Missing {}".format(set(self.missing_video_names)))
        if self.dset_name == "anet":
            self.data = [e for e in self.data if e["name"][2:] not in self.missing_video_names]
        else:
            self.data = [e for e in self.data if e["name"] not in self.missing_video_names]

    def _load_duration(self):
        """https://github.com/salesforce/densecap/blob/master/data/anet_dataset.py#L120
        Since the features are extracted not at the exact 0.5 secs. To get the real time for each feature,
        use `(idx + 1) * frame_to_second[vid_name] `
        """
        frame_to_second = {}
        sampling_sec = 0.5  # hard coded, only support 0.5
        if self.dset_name == "anet":
            with open(self.duration_file, "r") as f:
                for line in f:
                    vid_name, vid_dur, vid_frame = [l.strip() for l in line.split(",")]
                    frame_to_second[vid_name] = float(vid_dur) * int(
                        float(vid_frame) * 1. / int(float(vid_dur)) * sampling_sec) * 1. / float(vid_frame)
                frame_to_second["_0CqozZun3U"] = sampling_sec  # a missing video in anet
        elif self.dset_name == "yc2":
            with open(self.duration_file, "r") as f:
                for line in f:
                    vid_name, vid_dur, vid_frame = [l.strip() for l in line.split(",")]
                    frame_to_second[vid_name] = float(vid_dur) * math.ceil(
                        float(vid_frame) * 1. / float(vid_dur) * sampling_sec) * 1. / float(vid_frame)  # for yc2
        else:
            raise NotImplementedError("Only support anet and yc2, got {}".format(self.dset_name))
        return frame_to_second

    def _load_data(self, data_path):
        logging.info("Loading data from {}".format(data_path))
        raw_data = load_json(data_path)
        data = []
        for k, line in tqdm(raw_data.items()):
            line["name"] = k
            line["timestamps"] = line["timestamps"][:self.max_n_sen]
            line["sentences"] = line["sentences"][:self.max_n_sen]
            data.append(line)

        singel_sentence_data = []
        for d in data:
            num_sen = min(self.max_n_sen, len(d["sentences"]))
            singel_sentence_data.extend([
                {
                    "duration": d["duration"],
                    "name": d["name"],
                    "timestamp": d["timestamps"][idx],
                    "sentence": d["sentences"][idx],
                    "idx":idx
                } for idx in range(num_sen)])
        self.data = singel_sentence_data

        logging.info("Loading complete! {} examples".format(len(self)))

    def convert_example_to_features(self, example):
        """example single snetence
        {"name": str,
         "duration": float,
         "timestamp": [st(float), ed(float)],
         "sentence": str
        } or
        {"name": str,
         "duration": float,
         "timestamps": list([st(float), ed(float)]),
         "sentences": list(str)
        }
        """
        name = example["name"]
        video_name = name[2:] if self.dset_name == "anet" else name
        feat_path_resnet = os.path.join(self.video_feature_dir, "{}_resnet.npy".format(video_name))
        feat_path_bn = os.path.join(self.video_feature_dir, "{}_bn.npy".format(video_name))
        region_path = os.path.join(self.region_feature_dir, "{}.npz".format(video_name))
        region_kg = self.region_kg_dir[video_name]
        video_feature = np.concatenate([np.load(feat_path_resnet), np.load(feat_path_bn)], axis=1)
        region_feature = np.load(region_path, allow_pickle=True)['x']
        rf_300d = np.load(self.region_feature_300dir+video_name+'.npy', allow_pickle=True)
        if video_name not in self.asr_path.keys() or 'asr' not in self.asr_path[video_name].keys():
            asr = []
        else:
            asr = self.asr_path[video_name]['asr'][example["idx"]]
        tags = np.load(region_path, allow_pickle=True)['scores']
        if self.recurrent:  # recurrent
            num_sen = len(example["sentences"])
            single_video_features = []
            single_video_meta = []
            for clip_idx in range(num_sen):
                cur_data, cur_meta = self.clip_sentence_to_feature(example["name"],
                                                                   example["timestamps"][clip_idx],
                                                                   example["sentences"][clip_idx],
                                                                   video_feature,
                                                                   region_feature,
                                                                   tags)
                single_video_features.append(cur_data)
                single_video_meta.append(cur_meta)
            return single_video_features, single_video_meta
        else:  # single sentence
            clip_dataloader = self.clip_sentence_to_feature_untied \
                if self.untied else self.clip_sentence_to_feature

            cur_data, cur_meta = clip_dataloader(example["name"],
                                                 example["timestamp"],
                                                 example["sentence"],
                                                 video_feature,
                                                 region_feature,
                                                 rf_300d,
                                                 tags,
                                                 region_kg,
                                                 asr)
            return cur_data, cur_meta

    def print_see(self, kg, score, sentence):
        kg_tokens = [self.idx2word[w[2]] for w in kg]
        print('---------------------')
        print(score)
        print(kg_tokens)
        print(sentence)

    def classify(self, sen):
        if sen[0] == 993:
            return 2
        elif sen[0] == 995:
            return 3
        else:
            return 4

    def choose_regionrank(self, kg, region_feature, sentence):
        region_feature = torch.Tensor(region_feature).unsqueeze(0)
        kg_embed = self.word_embeddings(kg)[:, 2]  # (n,300)
        score_region = F.cosine_similarity(region_feature, kg_embed, dim=1)
        paixu, xuhao = torch.sort(score_region, descending=True)
        xuhao = xuhao.unsqueeze(-1).expand(kg.size(0), 3)
        kg = torch.gather(kg, 0, xuhao).tolist()
        kg = kg[:4]
        return kg

    def choose_asrrank(self, kgs_asr, kgs_out, asr_tokens,region_feature,sentence):
        new_kg = [w for w in kgs_asr if w[0] != 0] + [w[:3] for w in kgs_out if w[0] != 0]
        kg = []
        for t in new_kg:
            if t not in kg:
                kg.append(t)
        if len(kg) == 0:
            return kg
        kg = torch.IntTensor(kg)
        end = asr_tokens.index('[EOS]')
        asr_tokens = asr_tokens[1:end]
        if asr_tokens == []:
            return self.choose_regionrank(kg, region_feature,sentence)
        asr = [self.word2idx[w] for w in asr_tokens if w in self.word2idx.keys()]
        kg_embed = self.word_embeddings(kg)[:,2]  # (n,300)
        asr_embed = self.word_embeddings(torch.IntTensor(asr))  # (m,300)
        score_region = F.cosine_similarity(torch.sum(asr_embed, dim=0).unsqueeze(0), kg_embed, dim=1)
        paixu, xuhao = torch.sort(score_region, descending=True)
        xuhao = xuhao.unsqueeze(-1).expand(kg.size(0), 3)
        kg = torch.gather(kg, 0, xuhao).tolist()
        kg = kg[:9]
        return kg

    def add_region_kg_n(self, region_kg, m, region_features, asr_tokens, sentence):
        l_mask = [0] * self.max_len + [0] * self.max_kg_len
        mask_kg = torch.zeros((self.max_len + self.max_kg_len, self.max_len + self.max_kg_len))
        kg_token = [0] * self.max_kg_len
        relation_type = [0] * self.max_kg_len
        l_m = torch.zeros(m + 1)
        for i in range(int(m / 5)):
            if m < 200:
                if i % 3 == 0:
                    l_m[1 + i * 5:(i + 1) * 5 + 1] = 1
            elif m < 300:
                if i % 4 == 0:
                    l_m[1 + i * 5:(i + 1) * 5 + 1] = 1
            elif m < 400:
                if i % 5 == 0:
                    l_m[1 + i * 5:(i + 1) * 5 + 1] = 1
            else:
                if i % 6 == 0:
                    l_m[1 + i * 5:(i + 1) * 5 + 1] = 1
        j = 0
        for i in range(1, m + 1):
            if l_m[i] == 0:
                continue
            word = self.idx2word[region_kg[i]]
            kgs_asr = []
            kgs_out = []
            if word in self.asrkg.keys():
                kgs_asr = self.asrkg[word]
            if word in self.outkg.keys():
                kgs_out = self.outkg[word]
            kgs = self.choose_asrrank(kgs_asr, kgs_out, asr_tokens, region_features[i], sentence)
            for sen in kgs:  # 4
                relation_type[j] = self.classify(sen)
                sen = sen[2:3]  # no relationship
                if sen[0] == 0:
                    break
                length = len(sen)

                kg_token[j:j + length] = sen
                l_mask[self.max_len + j:self.max_len + j + length] = [1] * length
                l_mask[i] = 1
                mask_kg[i, self.max_len + j:self.max_len + j + length] = 1
                mask_kg[self.max_len + j:self.max_len + j + length, i] = 1
                mask_kg[self.max_len + j:self.max_len + j + length, self.max_len + j:self.max_len + j + length] = 1
                j = j + length
                if j > self.max_kg_len - 3:
                    gu = mask_kg.clone()
                    gu[self.max_len:self.max_len + j, 1:m + 1] = 1
                    print(m)
                    return kg_token, mask_kg, l_mask, gu, relation_type
        gu = mask_kg.clone()
        gu[self.max_len:self.max_len + j, 1:m + 1] = 1
        return kg_token, mask_kg, l_mask, gu, relation_type




    def clip_sentence_to_feature(self, name, timestamp, sentence, video_feature, region_fature, rf_300d, tags,region_kgs,asr):
        """ make features for a single clip-sentence pair.
        [CLS], [VID], ..., [VID], [SEP], [BOS], [WORD], ..., [WORD], [EOS]
        Args:
            name: str,
            timestamp: [float, float]
            sentence: str
            video_feature: np array
        """
        frm2sec = self.frame_to_second[name[2:]] if self.dset_name == "anet" else self.frame_to_second[name]

        feat, region, rf, tag, region_kg,video_tokens, region_token, video_mask, region_mask,m = self._load_indexed_video_feature(video_feature, region_fature, rf_300d,tags,region_kgs,timestamp, frm2sec)

        text_tokens, text_mask = self._tokenize_pad_sentence(sentence, self.max_t_len)

        if asr == []:
            asr_tokens, asr_mask = [self.BOS_TOKEN] + [self.EOS_TOKEN] + ['[PAD]']*(self.max_asr_len-2), [0]*self.max_asr_len
        else:
            asr_tokens,asr_mask = self._tokenize_pad_sentence(asr, self.max_asr_len)

        kg_token, kg_mask, l, gu, relation_type = self.add_region_kg_n(region_kg, m, rf, asr_tokens,sentence)

        input_tokens = video_tokens + asr_tokens + text_tokens
        input_tokens2 = region_token + asr_tokens + text_tokens + kg_token

        input_ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in input_tokens]
        input_ids2 = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in input_tokens2]

        # shifted right, `-1` is ignored when calculating CrossEntropy Loss
        input_labels = [self.IGNORE if m == 0 else tid for tid, m in zip(input_ids[-len(text_mask):], text_mask)][1:] + [self.IGNORE]

        input_mask = video_mask + asr_mask + text_mask
        input_mask2 = region_mask + asr_mask + text_mask + [0]*self.max_kg_len
        token_type_ids = [0] * self.max_v_len + [1]*self.max_asr_len + [1] * self.max_t_len
        token_type_ids2 = [0] * (98*5+2) + [1]*self.max_asr_len + [1] * self.max_t_len + relation_type

        data = dict(
            name=name,
            input_tokens=input_tokens,
            input_tokens2=input_tokens2,
            # model inputs
            input_ids=np.array(input_ids).astype(np.int64),
            input_ids2=np.array(input_ids2).astype(np.int64),
            input_labels=np.array(input_labels).astype(np.int64),
            input_mask=np.array(input_mask).astype(np.float32),
            input_mask2=np.array(input_mask2).astype(np.float32),
            token_type_ids=np.array(token_type_ids).astype(np.int64),
            token_type_ids2=np.array(token_type_ids2).astype(np.int64),
            video_feature=feat.astype(np.float32),
            region_feature=region.astype(np.float32),
            tag=np.array(tag).astype(np.int64),
            kg_mask=np.array(kg_mask).astype(np.int64),
            l_mask =np.array(l).astype(np.int64),
            gu = np.array(gu).astype(np.int64)
        )
        meta = dict(
            # meta
            name=name,
            timestamp=timestamp,
            sentence=sentence
        )
        return data, meta

    def clip_sentence_to_feature_untied(self, name, timestamp, sentence, raw_video_feature):
        """ make features for a single clip-sentence pair.
        [CLS], [VID], ..., [VID], [SEP], [BOS], [WORD], ..., [WORD], [EOS]
        Args:
            name: str,
            timestamp: [float, float]
            sentence: str
            raw_video_feature: np array, N x D, for the whole video
        """
        frm2sec = self.frame_to_second[name[2:]] if self.dset_name == "anet" else self.frame_to_second[name]

        # video + text tokens
        video_feature, video_mask = self._load_indexed_video_feature_untied(raw_video_feature, timestamp, frm2sec)
        text_tokens, text_mask = self._tokenize_pad_sentence(sentence)

        text_ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in text_tokens]
        # shifted right, `-1` is ignored when calculating CrossEntropy Loss
        text_labels = [self.IGNORE if m == 0 else tid for tid, m in zip(text_ids, text_mask)][1:] + [self.IGNORE]

        data = dict(
            name=name,
            text_tokens=text_tokens,
            # model inputs
            text_ids=np.array(text_ids).astype(np.int64),
            text_mask=np.array(text_mask).astype(np.float32),
            text_labels=np.array(text_labels).astype(np.int64),
            video_feature=video_feature.astype(np.float32),
            video_mask=np.array(video_mask).astype(np.float32),
        )
        meta = dict(
            # meta
            name=name,
            timestamp=timestamp,
            sentence=sentence,
        )
        return data, meta

    @classmethod
    def _convert_to_feat_index_st_ed(cls, feat_len, timestamp, frm2sec):
        """convert wall time st_ed to feature index st_ed"""
        st = int(math.floor(timestamp[0] / frm2sec))
        ed = int(math.ceil(timestamp[1] / frm2sec))
        ed = min(ed, feat_len-1)
        st = min(st, ed-1)
        assert st <= ed <= feat_len, "st {} <= ed {} <= feat_len {}".format(st, ed, feat_len)
        return st, ed

    def _load_indexed_video_feature(self, raw_feat, region_feat, rf_300d, tags, region_kgs,timestamp, frm2sec):
        """ [CLS], [VID], ..., [VID], [SEP], [PAD], ..., [PAD],
        All non-PAD tokens are valid, will have a mask value of 1.
        Returns:
            feat is padded to length of (self.max_v_len + self.max_t_len,)
            video_tokens: self.max_v_len
            mask: self.max_v_len
        """
        if raw_feat.shape[0]>region_feat.shape[0]:
            raw_feat = raw_feat[:region_feat.shape[0]]
            if timestamp[1]>region_feat.shape[0]:
                timestamp[1] = region_feat.shape[0]

        region_feat = region_feat.reshape(-1,2048)
        tags = tags.reshape(-1)

        region_kgs = np.array(region_kgs).reshape(-1)
        max_v_l = self.max_v_len - 2

        feat_len = len(raw_feat)
        st, ed = self._convert_to_feat_index_st_ed(feat_len, timestamp, frm2sec)
        indexed_feat_len = ed - st + 1

        feat = np.zeros((self.max_v_len + self.max_asr_len + self.max_t_len, raw_feat.shape[1]))  # includes [CLS], [SEP]
        region = np.zeros((max_v_l*5+2 + self.max_asr_len + self.max_t_len+self.max_kg_len, 2048))
        rf = np.zeros((max_v_l * 5 + 2 + self.max_asr_len + self.max_t_len + self.max_kg_len, 300))
        tag = np.zeros((max_v_l*5+2 + self.max_asr_len + self.max_t_len+self.max_kg_len))
        region_kg = np.zeros(max_v_l*5+2+ self.max_asr_len + self.max_t_len+ self.max_kg_len)

        if indexed_feat_len > max_v_l:
            downsamlp_indices = np.linspace(st, ed, max_v_l, endpoint=True).astype(np.int).tolist()
            assert max(downsamlp_indices) < feat_len
            feat[1:max_v_l+1] = raw_feat[downsamlp_indices]  # truncate, sample???

            downsamlp_indices = [i*5 for i in downsamlp_indices]
            downsamlp_indices1 = downsamlp_indices + [i+1 for i in downsamlp_indices] + [i+2 for i in downsamlp_indices] + [i+3 for i in downsamlp_indices] + [i+4 for i in downsamlp_indices]
            downsamlp_indices1 = np.sort(downsamlp_indices1)

            region[1:max_v_l*5+1] = region_feat[downsamlp_indices1]   #!!!
            rf[1:max_v_l*5+1] = rf_300d[downsamlp_indices1]
            tag[1:max_v_l*5+1] = tags[downsamlp_indices1]
            region_kg[1:max_v_l*5+1] = region_kgs[downsamlp_indices1]

            video_tokens = [self.CLS_TOKEN] + [self.VID_TOKEN] * max_v_l + [self.SEP_TOKEN]
            region_tokens = [self.CLS_TOKEN] + [self.VID_TOKEN] * max_v_l*5 + [self.SEP_TOKEN]
            mask_video = [1] * (max_v_l + 2)
            mask_region = [1] * (max_v_l*5 + 2)
            m = max_v_l*5
        else:
            valid_l = ed - st + 1
            feat[1:valid_l+1] = raw_feat[st:ed + 1]
            region[1:valid_l*5+1] = region_feat[st*5:st*5+valid_l*5]   #！！！
            rf[1:valid_l * 5 + 1] = rf_300d[st * 5:st * 5 + valid_l * 5]
            tag[1:valid_l*5+1] = tags[st*5:st*5+valid_l*5]
            region_kg[1:valid_l*5+1] = region_kgs[st*5:st*5+valid_l*5]
            video_tokens = [self.CLS_TOKEN] + [self.VID_TOKEN] * valid_l + [self.SEP_TOKEN] + [self.PAD_TOKEN] * (max_v_l - valid_l)
            region_tokens = [self.CLS_TOKEN] + [self.VID_TOKEN] * valid_l*5 + [self.SEP_TOKEN] + [self.PAD_TOKEN] * (max_v_l - valid_l)*5
            mask_video = [1] * (valid_l + 2) + [0] * (max_v_l - valid_l)
            mask_region = [1] * (valid_l*5 + 2) + [0] * (max_v_l - valid_l)*5
            m = valid_l*5
        return feat, region, rf, tag, region_kg, video_tokens, region_tokens, mask_video, mask_region, m

    def _load_indexed_video_feature_untied(self, raw_feat, timestamp, frm2sec):
        """ Untied version: [VID], ..., [VID], [PAD], ..., [PAD], len == max_v_len
        Returns:
            feat is padded to length of (self.max_v_len,)
            mask: self.max_v_len, with 1 indicates valid bits, 0 indicates padding
        """
        max_v_l = self.max_v_len
        feat_len = len(raw_feat)
        st, ed = self._convert_to_feat_index_st_ed(feat_len, timestamp, frm2sec)
        indexed_feat_len = ed - st + 1

        if indexed_feat_len > max_v_l:
            downsamlp_indices = np.linspace(st, ed, max_v_l, endpoint=True).astype(np.int).tolist()
            assert max(downsamlp_indices) < feat_len
            feat = raw_feat[downsamlp_indices]  # truncate, sample???
            mask = [1] * max_v_l  # no padding
        else:
            feat = np.zeros((max_v_l, raw_feat.shape[1]))  # only video features and padding
            valid_l = ed - st + 1
            feat[:valid_l] = raw_feat[st:ed + 1]
            mask = [1] * valid_l + [0] * (max_v_l - valid_l)
        return feat, mask

    def _tokenize_pad_sentence(self, sentence,max_t_len):
        """[BOS], [WORD1], [WORD2], ..., [WORDN], [EOS], [PAD], ..., [PAD], len == max_t_len
        All non-PAD values are valid, with a mask value of 1
        """
        temp = nltk.tokenize.word_tokenize(sentence.lower())
        # if len(temp)>300:
            # print(len(temp))
        sentence_tokens = temp[:max_t_len - 2]
        sentence_tokens = [self.BOS_TOKEN] + sentence_tokens + [self.EOS_TOKEN]

        # pad
        valid_l = len(sentence_tokens)
        mask = [1] * valid_l + [0] * (max_t_len - valid_l)
        sentence_tokens += [self.PAD_TOKEN] * (max_t_len - valid_l)
        return sentence_tokens, mask

    def convert_ids_to_sentence(self, ids, rm_padding=True, return_sentence_only=True):
        """A list of token ids"""
        rm_padding = True if return_sentence_only else rm_padding
        if rm_padding:
            raw_words = [self.idx2word[wid] for wid in ids if wid not in [self.PAD, self.IGNORE]]
        else:
            raw_words = [self.idx2word[wid] for wid in ids if wid != self.IGNORE]

        # get only sentences, the tokens between `[BOS]` and the first `[EOS]`
        if return_sentence_only:
            words = []
            for w in raw_words[1:]:  # no [BOS]
                if w != self.EOS_TOKEN:
                    words.append(w)
                else:
                    break
        else:
            words = raw_words
        return " ".join(words)


def prepare_batch_inputs(batch, device, non_blocking=False):
    batch_inputs = dict()
    bsz = len(batch["name"])
    for k, v in batch.items():
        assert bsz == len(v), (bsz, k, v)
        if isinstance(v, torch.Tensor):
            batch_inputs[k] = v.to(device, non_blocking=non_blocking)
        else:  # all non-tensor values
            batch_inputs[k] = v
    return batch_inputs


def step_collate(padded_batch_step):
    """The same step (clip-sentence pair) from each example"""
    c_batch = dict()
    for key in padded_batch_step[0]:
        value = padded_batch_step[0][key]
        if isinstance(value, list):
            c_batch[key] = [d[key] for d in padded_batch_step]
        else:
            c_batch[key] = default_collate([d[key] for d in padded_batch_step])
    return c_batch


def caption_collate(batch):
    """get rid of unexpected list transpose in default_collate
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py#L66

    HOW to batch clip-sentence pair?
    1) directly copy the last sentence, but do not count them in when back-prop OR
    2) put all -1 to their text token label, treat
    """
    # collect meta
    raw_batch_meta = [e[1] for e in batch]
    batch_meta = []
    for e in raw_batch_meta:
        cur_meta = dict(
            name=None,
            timestamp=[],
            gt_sentence=[]
        )
        for d in e:
            cur_meta["name"] = d["name"]
            cur_meta["timestamp"].append(d["timestamp"])
            cur_meta["gt_sentence"].append(d["sentence"])
        batch_meta.append(cur_meta)

    batch = [e[0] for e in batch]
    # Step1: pad each example to max_n_sen
    max_n_sen = max([len(e) for e in batch])
    raw_step_sizes = []

    padded_batch = []
    padding_clip_sen_data = copy.deepcopy(batch[0][0])  # doesn"t matter which one is used
    padding_clip_sen_data["input_labels"][:] = RecursiveCaptionDataset.IGNORE
    for ele in batch:
        cur_n_sen = len(ele)
        if cur_n_sen < max_n_sen:
            ele = ele + [padding_clip_sen_data] * (max_n_sen - cur_n_sen)
        raw_step_sizes.append(cur_n_sen)
        padded_batch.append(ele)

    # Step2: batching each steps individually in the batches
    collated_step_batch = []
    for step_idx in range(max_n_sen):
        collated_step = step_collate([e[step_idx] for e in padded_batch])
        collated_step_batch.append(collated_step)
    return collated_step_batch, raw_step_sizes, batch_meta


def single_sentence_collate(batch):
    """get rid of unexpected list transpose in default_collate
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py#L66
    """
    # collect meta
    batch_meta = [{"name": e[1]["name"],
                   "timestamp": e[1]["timestamp"],
                   "gt_sentence": e[1]["sentence"]
                   } for e in batch]  # change key
    padded_batch = step_collate([e[0] for e in batch])
    return padded_batch, None, batch_meta
