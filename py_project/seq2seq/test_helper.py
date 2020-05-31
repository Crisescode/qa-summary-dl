# -*- coding:utf-8 -*-

import tensorflow as tf


class Hypothesis:
    """ Class designed to hold hypothesises throughout the beamSearch decoding """

    def __init__(self, tokens, log_probs, hidden, attn_dists):
        self.tokens = tokens  # list of all the tokens from time 0 to the current time step t
        self.log_probs = log_probs  # list of the log probabilities of the tokens of the tokens
        self.hidden = hidden  # decoder hidden state after the last token decoding
        self.attn_dists = attn_dists  # attention dists of all the tokens
        self.abstract = ""

    def extend(self, token, log_prob, hidden, attn_dist):
        """Method to extend the current hypothesis by adding the next decoded token and all the informations associated with it"""
        return Hypothesis(tokens=self.tokens + [token],  # we add the decoded token
                          log_probs=self.log_probs + [log_prob],  # we add the log prob of the decoded token
                          hidden=hidden,  # we update the state
                          attn_dists=self.attn_dists + [attn_dist])

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def tot_log_prob(self):
        return sum(self.log_probs)

    @property
    def avg_log_prob(self):
        return self.tot_log_prob / len(self.tokens)


def beam_decode(model, batch, vocab, params):
    # 初始化mask
    start_index = vocab['<START>']
    stop_index = vocab['<STOP>']

    batch_size = params['batch_size']

    # 单步decoder
    def decoder_onestep(enc_output, dec_input, dec_hidden):
        # 单个时间步 运行
        preds, dec_hidden, context_vector, attention_weights = model.call_decoder_onestep(dec_input, dec_hidden,
                                                                                          enc_output)
        # 拿到top k个index 和 概率
        top_k_probs, top_k_ids = tf.nn.top_k(tf.squeeze(preds), k=params["beam_size"])
        # 计算log概率
        top_k_log_probs = tf.math.log(top_k_probs)
        # 返回需要保存的中间结果和概率
        return preds, dec_hidden, context_vector, attention_weights, top_k_log_probs, top_k_ids

    # 计算第encoder的输出
    enc_output, enc_hidden = model.call_encoder(batch)

    # 初始化batch size个 假设对象
    hyps = [Hypothesis(tokens=[start_index],
                       log_probs=[0.0],
                       hidden=enc_hidden[0],
                       attn_dists=[],
                       ) for _ in range(batch_size)]
    # 初始化结果集
    results = []  # list to hold the top beam_size hypothesises
    # 遍历步数
    steps = 0  # initial step

    # 第一个decoder输入 开始标签
    dec_input = tf.expand_dims([start_index] * batch_size, 1)
    # 第一个隐藏层输入
    dec_hidden = enc_hidden

    # 长度还不够 并且 结果还不够 继续搜索
    while steps < params['max_dec_steps'] and len(results) < params['beam_size']:
        # 获取最新待使用的token
        latest_tokens = [h.latest_token for h in hyps]
        # 获取所以隐藏层状态
        hiddens = [h.hidden for h in hyps]
        # 单步运行decoder 计算需要的值
        preds, dec_hidden, context_vector, attention_weights, top_k_log_probs, top_k_ids = decoder_onestep(enc_output,
                                                                                                           dec_input,
                                                                                                           dec_hidden)

        # 现阶段全部可能情况
        all_hyps = []
        # 原有的可能情况数量
        num_orig_hyps = 1 if steps == 0 else len(hyps)

        # 遍历添加所有可能结果
        for i in range(num_orig_hyps):
            h, new_hidden, attn_dist = hyps[i], dec_hidden[i], attention_weights[i]
            # 分裂 添加 beam size 种可能性
            for j in range(params['beam_size']):
                # 构造可能的情况
                new_hyp = h.extend(token=top_k_ids[i, j].numpy(),
                                   log_prob=top_k_log_probs[i, j],
                                   hidden=new_hidden,
                                   attn_dist=attn_dist)
                # 添加可能情况
                all_hyps.append(new_hyp)

        # 重置
        hyps = []
        # 按照概率来排序
        sorted_hyps = sorted(all_hyps, key=lambda h: h.avg_log_prob, reverse=True)

        # 筛选top前beam_size句话
        for h in sorted_hyps:
            if h.latest_token == stop_index:
                # 长度符合预期,遇到句尾,添加到结果集
                if steps >= params['min_dec_steps']:
                    results.append(h)
            else:
                # 未到结束 ,添加到假设集
                hyps.append(h)

            # 如果假设句子正好等于beam_size 或者结果集正好等于beam_size 就不在添加
            if len(hyps) == params['beam_size'] or len(results) == params['beam_size']:
                break

        steps += 1

    if len(results) == 0:
        results = hyps

    hyps_sorted = sorted(results, key=lambda h: h.avg_log_prob, reverse=True)
    best_hyp = hyps_sorted[0]
    # best_hyp.abstract = " ".join([reverse_vocab[index] for index in best_hyp.tokens])
    return best_hyp
