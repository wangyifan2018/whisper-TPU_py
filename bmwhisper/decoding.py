from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import time
import numpy as np
from scipy.special import softmax
from .utils import CHUNK_LENGTH
from .tokenizer import Tokenizer, get_tokenizer
from .utils import compression_ratio, fp16_cast, uint16_to_fp16, log_softmax, logsumexp
if TYPE_CHECKING:
    from .model import Whisper


def detect_language(
    model: "Whisper", mel: np.ndarray, tokenizer: Tokenizer = None
) -> Tuple[np.ndarray, List[dict]]:
    """
    Detect the spoken language in the audio, and return them as list of strings, along with the ids
    of the most probable language tokens and the probability distribution over all language tokens.
    This is performed outside the main decode loop in order to not interfere with kv-caching.

    Returns
    -------
    language_tokens : np.ndarray, shape = (n_audio,)
        ids of the most probable language tokens, which appears after the startoftranscript token.
    language_probs : List[Dict[str, float]], length = n_audio
        list of dictionaries containing the probability distribution over all languages.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer(model.is_multilingual)
    if (
        tokenizer.language is None
        or tokenizer.language_token not in tokenizer.sot_sequence
    ):
        raise ValueError(
            "This model doesn't have language tokens so it can't perform lang id"
        )

    single = mel.ndim == 2
    if single:
        mel = np.expand_dims(mel, 0)

    start_time = time.time()
    # skip encoder forward pass if already-encoded audio features were given
    if mel.shape[-2:] != (model.dims.n_audio_ctx, model.dims.n_audio_state):
        # transform type from encoder inputs
        # type
        mel = mel.astype(np.float16)
        mel = mel if mel.flags.c_contiguous else np.ascontiguousarray(mel)

        model.encoder_input_tensors_map[model.combined_whisper_engine.get_input_names(model.encoder_engine_graph_name)[0]].update_data(fp16_cast(mel));

        model.combined_whisper_engine.process(model.encoder_engine_graph_name,model.encoder_input_tensors_map,model.encoder_output_tensors_map)
        mel_out_tensor = list(model.encoder_output_tensors_map.values())[0]
        mel_out = uint16_to_fp16(mel_out_tensor.asnumpy())

        model.time += time.time() - start_time
        model.call_encoder += 1

    # forward pass using a single token, startoftranscript
    n_audio = mel_out.shape[0]
    x = np.array([[tokenizer.sot]] * n_audio)  # [n_audio, 1]
    start_time = time.time()
    logits = model.logits(x, mel_out)[:, 0].astype(np.float32)

    # collect detected languages; suppress all non-language tokens
    mask = np.ones(logits.shape[-1], dtype=bool)
    mask[list(tokenizer.all_language_tokens)] = False
    logits[:, mask] = -np.inf
    language_tokens = np.argmax(logits, axis=-1)
    language_token_probs = softmax(logits, axis=-1)
    language_probs = [
        {
            c: language_token_probs[i, j].item()
            for j, c in zip(tokenizer.all_language_tokens, tokenizer.all_language_codes)
        }
        for i in range(n_audio)
    ]

    if single:
        language_tokens = language_tokens[0]
        language_probs = language_probs[0]

    return language_tokens, language_probs


@dataclass(frozen=True)
class DecodingOptions:
    # whether to perform X->X "transcribe" or X->English "translate"
    task: str = "transcribe"

    # language that the audio is in; uses detected language if None
    language: Optional[str] = None

    # sampling-related options
    temperature: float = 0.0
    sample_len: Optional[int] = None  # maximum number of tokens to sample
    best_of: Optional[int] = None  # number of independent sample trajectories, if t > 0
    beam_size: Optional[int] = None  # number of beams in beam search, if t == 0
    patience: Optional[float] = None  # patience in beam search (arxiv:2204.05424)

    # "alpha" in Google NMT, or None for length norm, when ranking generations
    # to select which to return among the beams or best-of-N samples
    length_penalty: Optional[float] = None

    # text or tokens to feed as the prompt or the prefix; for more info:
    # https://github.com/openai/whisper/discussions/117#discussioncomment-3727051
    prompt: Optional[Union[str, List[int]]] = None  # for the previous context
    prefix: Optional[Union[str, List[int]]] = None  # to prefix the current context

    # list of tokens ids (or comma-separated token ids) to suppress
    # "-1" will suppress a set of symbols as defined in `tokenizer.non_speech_tokens()`
    suppress_tokens: Optional[Union[str, Iterable[int]]] = "-1"
    suppress_blank: bool = True  # this will suppress blank outputs

    # timestamp sampling options
    without_timestamps: bool = False  # use <|notimestamps|> to sample text tokens only
    max_initial_timestamp: Optional[float] = 1.0

    # implementation details
    padding_size: int = 448 # max pre-allocation of key-value cache

@dataclass(frozen=True)
class DecodingResult:
    audio_features: np.ndarray
    language: str
    language_probs: Optional[Dict[str, float]] = None
    tokens: List[int] = field(default_factory=list)
    text: str = ""
    avg_logprob: float = np.nan
    no_speech_prob: float = np.nan
    temperature: float = np.nan
    compression_ratio: float = np.nan


class PyTorchInference():
    def __init__(self, model: "Whisper", initial_token_length: int):
        self.model: "Whisper" = model

    def rearrange_kv_cache(
            self,
            source_indices
        ):
        if source_indices != list(range(len(source_indices))):
            start_time = time.time()
            indices = np.array(source_indices, dtype=np.int32)
            indices = indices if indices.flags.contiguous else indices.copy()

            self.model.kvcache_rearrange_input_list[0][self.model.kvcache_rearrange_input_names[1]].update_data(indices)
            for i in range(2 * self.model.dims.n_text_layer):
                self.model.combined_whisper_engine.process(self.model.kvcache_rearrange_graph_name, self.model.kvcache_rearrange_input_list[i], self.model.kvcache_rearrange_output_list[i])

            self.model.time += time.time() - start_time
            self.model.call_kvcache_rearrange += 2 * self.model.dims.n_text_layer
            return

class SequenceRanker:
    def rank(
        self, tokens: List[List[np.ndarray]], sum_logprobs: List[List[float]]
    ) -> List[int]:
        """
        Given a list of groups of samples and their cumulative log probabilities,
        return the indices of the samples in each group to select as the final result
        """
        raise NotImplementedError

class MaximumLikelihoodRanker(SequenceRanker):
    """
    Select the sample with the highest log probabilities, penalized using either
    a simple length normalization or Google NMT paper's length penalty
    """
    def __init__(self, length_penalty: Optional[float]):
        self.length_penalty = length_penalty

    def rank(self, tokens: List[List[np.ndarray]], sum_logprobs: List[List[float]]):
        def scores(logprobs, lengths):
            result = []
            for logprob, length in zip(logprobs, lengths):
                if self.length_penalty is None:
                    penalty = length
                else:
                    # from the Google NMT paper
                    penalty = ((5 + length) / 6) ** self.length_penalty
                result.append(logprob / penalty)
            return result

        # get the sequence with the highest score
        lengths = [[len(t) for t in s] for s in tokens]
        return [np.argmax(scores(p, l)) for p, l in zip(sum_logprobs, lengths)]

class GreedyDecoder():
    def __init__(self, temperature: float, eot: int):
        self.temperature = temperature
        self.eot = eot

    def update(
        self,
        tokens: np.ndarray,
        logits: np.ndarray,
        sum_logprobs: np.ndarray
    ):
        if self.temperature == 0:
            next_tokens = logits.argmax(axis=-1)
        else:
            # 分布式抽样
            probs = np.exp(logits / self.temperature)
            probs /= np.sum(probs, axis=-1, keepdims=True)
            next_tokens = np.array([np.random.choice(len(p), p=p) for p in probs])

        logprobs = log_softmax(logits)
        # 选择对应的 logprobs
        current_logprobs = logprobs[np.arange(logprobs.shape[0]), next_tokens]
        sum_logprobs += current_logprobs * (tokens[:, -1] != self.eot)

        next_tokens[tokens[:, -1] == self.eot] = self.eot
        tokens = np.concatenate([tokens, next_tokens[:, None]], axis=-1)

        completed = np.all(tokens[:, -1] == self.eot)
        return tokens, completed

    def finalize(self, tokens: np.ndarray, sum_logprobs: np.ndarray):
        tokens = np.pad(tokens, ((0, 0), (0, 1)), constant_values=self.eot)
        return tokens, sum_logprobs.tolist()

class BeamSearchDecoder():
    def __init__(
        self,
        beam_size: int,
        eot: int,
        inference,
        patience: Optional[float] = None,
    ):
        self.beam_size = beam_size
        self.eot = eot
        self.inference = inference
        self.patience = patience or 1.0
        self.max_candidates: int = round(beam_size * self.patience)
        self.finished_sequences = None

        assert (
            self.max_candidates > 0
        ), f"Invalid beam size ({beam_size}) or patience ({patience})"

    def reset(self):
        self.finished_sequences = None

    def update(
        self,
        tokens: np.ndarray,
        logits: np.ndarray,
        sum_logprobs: np.ndarray,
        self_attention_kcache: np.ndarray = None,
        self_attention_vcache: np.ndarray = None,
    ) -> Tuple[np.ndarray, bool]:
        if tokens.shape[0] % self.beam_size != 0:
            raise ValueError(f"{tokens.shape}[0] % {self.beam_size} != 0")

        n_audio = tokens.shape[0] // self.beam_size
        if self.finished_sequences is None:  # for the first update
            self.finished_sequences = [{} for _ in range(n_audio)]

        logprobs = log_softmax(logits.astype(np.float32), axis=-1)
        next_tokens, source_indices, finished_sequences = [], [], []
        for i in range(n_audio):
            scores, sources, finished = {}, {}, {}

            # STEP 1: calculate the cumulative log probabilities for possible candidates
            for j in range(self.beam_size):
                idx = i * self.beam_size + j
                prefix = tokens[idx].tolist()
                topk_indices = np.argsort(logprobs[idx])[-(self.beam_size + 1):][::-1]
                topk_values = logprobs[idx][topk_indices]
                for logprob, token_index in zip(topk_values, topk_indices):
                    new_logprob = (sum_logprobs[idx] + logprob).item()
                    token = token_index.item()
                    sequence = tuple(prefix + [token])
                    scores[sequence] = new_logprob
                    sources[sequence] = idx

            # STEP 2: rank the candidates and keep the top beam_size sequences for each audio
            saved = 0
            for sequence in sorted(scores, key=scores.get, reverse=True):
                if sequence[-1] == self.eot:
                    finished[sequence] = scores[sequence]
                else:
                    sum_logprobs[len(next_tokens)] = scores[sequence]
                    next_tokens.append(sequence)
                    source_indices.append(sources[sequence])

                    saved += 1
                    if saved == self.beam_size:
                        break

            finished_sequences.append(finished)

        tokens = np.array(next_tokens)

        if self_attention_kcache:
            self.inference.rearrange_kv_cache(
                source_indices,
                self_attention_kcache,
                self_attention_vcache,
            )
        else:
            self.inference.rearrange_kv_cache(source_indices)

        # add newly finished sequences to self.finished_sequences
        assert len(self.finished_sequences) == len(finished_sequences)
        for previously_finished, newly_finished in zip(
            self.finished_sequences, finished_sequences
        ):
            for seq in sorted(newly_finished, key=newly_finished.get, reverse=True):
                if len(previously_finished) >= self.max_candidates:
                    break  # the candidate list is full
                previously_finished[seq] = newly_finished[seq]

        # mark as completed if all audio has enough number of samples
        completed = all(
            len(sequences) >= self.max_candidates
            for sequences in self.finished_sequences
        )
        return tokens, completed

    def finalize(self, preceding_tokens: np.ndarray, sum_logprobs: np.ndarray):
        """Finalize search and return the final candidate sequences

        Parameters
        ----------
        tokens : np.ndarray, shape = (n_audio, n_group, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence

        sum_logprobs : np.ndarray, shape = (n_audio, n_group)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Sequence[Sequence[np.ndarray]], length = n_audio
            sequence of np.ndarrays containing candidate token sequences, for each audio input

        sum_logprobs : List[List[float]], length = n_audio
            sequence of cumulative log probabilities corresponding to the above

        """

        # collect all finished sequences, including patience, and add unfinished ones if not enough
        sum_logprobs = sum_logprobs
        for i, sequences in enumerate(self.finished_sequences):
            if (
                len(sequences) < self.beam_size
            ):  # when not enough sequences are finished
                for j in list(np.argsort(sum_logprobs[i]))[::-1]:
                    sequence = preceding_tokens[i, j].tolist() + [self.eot]
                    sequences[tuple(sequence)] = sum_logprobs[i][j].item()
                    if len(sequences) >= self.beam_size:
                        break

        tokens: List[List[np.ndarray]] = [
            [np.array(seq) for seq in sequences.keys()]
            for sequences in self.finished_sequences
        ]
        sum_logprobs: List[List[float]] = [
            list(sequences.values()) for sequences in self.finished_sequences
        ]
        return tokens, sum_logprobs


class LogitFilter:
    def apply(self, logits: np.ndarray, tokens: np.ndarray) -> None:
        """Apply any filtering or masking to logits in-place

        Parameters
        ----------
        logits : np.ndarray, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        tokens : np.ndarray, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        """
        raise NotImplementedError


class SuppressBlank(LogitFilter):
    def __init__(self, tokenizer: Tokenizer, sample_begin: int):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin

    def apply(self, logits: np.ndarray, tokens: np.ndarray):
        if tokens.shape[1] == self.sample_begin:
            logits[:, self.tokenizer.encode(" ") + [self.tokenizer.eot]] = -np.inf


class SuppressTokens(LogitFilter):
    def __init__(self, suppress_tokens: Sequence[int]):
        self.suppress_tokens = list(suppress_tokens)

    def apply(self, logits: np.ndarray, tokens: np.ndarray):
        logits[:, self.suppress_tokens] = -np.inf


class ApplyTimestampRules(LogitFilter):
    def __init__(
        self,
        tokenizer: Tokenizer,
        sample_begin: int,
        max_initial_timestamp_index: Optional[int],
    ):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin
        self.max_initial_timestamp_index = max_initial_timestamp_index

    def apply(self, logits: np.ndarray, tokens: np.ndarray):
        # suppress <|notimestamps|> which is handled by without_timestamps
        if self.tokenizer.no_timestamps is not None:
            logits[:, self.tokenizer.no_timestamps] = -np.inf

        # timestamps have to appear in pairs, except directly before EOT; mask logits accordingly
        for k in range(tokens.shape[0]):
            sampled_tokens = tokens[k, self.sample_begin :]
            seq = [t for t in sampled_tokens.tolist()]
            last_was_timestamp = (
                len(seq) >= 1 and seq[-1] >= self.tokenizer.timestamp_begin
            )
            penultimate_was_timestamp = (
                len(seq) < 2 or seq[-2] >= self.tokenizer.timestamp_begin
            )

            if last_was_timestamp:
                if penultimate_was_timestamp:  # has to be non-timestamp
                    logits[k, self.tokenizer.timestamp_begin :] = -np.inf
                else:  # cannot be normal text tokens
                    logits[k, : self.tokenizer.eot] = -np.inf

            mask = sampled_tokens >= self.tokenizer.timestamp_begin
            timestamps = sampled_tokens[mask]
            if timestamps.size > 0:
                # timestamps shouldn't decrease; forbid timestamp tokens smaller than the last
                # also force each segment to have a nonzero length, to prevent infinite looping
                if last_was_timestamp and not penultimate_was_timestamp:
                    timestamp_last = timestamps[-1]
                else:
                    timestamp_last = timestamps[-1] + 1
                logits[k, self.tokenizer.timestamp_begin : timestamp_last] = -np.inf

        if tokens.shape[1] == self.sample_begin:
            # suppress generating non-timestamp tokens at the beginning
            logits[:, : self.tokenizer.timestamp_begin] = -np.inf

            # apply the `max_initial_timestamp` option
            if self.max_initial_timestamp_index is not None:
                last_allowed = (
                    self.tokenizer.timestamp_begin + self.max_initial_timestamp_index
                )
                logits[:, last_allowed + 1 :] = -np.inf

        # if sum of probability over timestamps is above any other token, sample timestamp
        logprobs = log_softmax(logits.astype(np.float32), axis=-1)
        for k in range(tokens.shape[0]):
            timestamp_logprob = logsumexp(logprobs[k, self.tokenizer.timestamp_begin:], axis=-1)
            max_text_token_logprob = logprobs[k, : self.tokenizer.timestamp_begin].max()
            if timestamp_logprob > max_text_token_logprob:
                logits[k, : self.tokenizer.timestamp_begin] = -np.inf


class DecodingTask:
    sequence_ranker: SequenceRanker
    logit_filters: List[LogitFilter]

    def __init__(self, model: "Whisper", options: DecodingOptions):
        self.model = model

        language = options.language or "en"
        tokenizer = get_tokenizer(
            model.is_multilingual, num_languages=model.num_languages, language=language, task=options.task
        )
        self.tokenizer: Tokenizer = tokenizer
        self.options: DecodingOptions = self._verify_options(options)

        self.n_group: int = options.beam_size or options.best_of or 1
        self.n_ctx: int = model.dims.n_text_ctx
        self.sample_len: int = options.sample_len or model.dims.n_text_ctx // 2
        self.n_text_head = self.model.dims.n_text_head
        self.n_text_layer = self.model.dims.n_text_layer
        self.padding_size = options.padding_size

        self.sot_sequence: Tuple[int] = tokenizer.sot_sequence
        if self.options.without_timestamps:
            self.sot_sequence = tokenizer.sot_sequence_including_notimestamps

        self.initial_tokens: Tuple[int] = self._get_initial_tokens()
        self.sample_begin: int = len(self.initial_tokens)
        self.sot_index: int = self.initial_tokens.index(tokenizer.sot)

        # inference: implements the forward pass through the decoder, including kv caching
        self.inference = PyTorchInference(model, len(self.initial_tokens))

        # sequence ranker: implements how to rank a group of sampled sequences
        self.sequence_ranker = MaximumLikelihoodRanker(options.length_penalty)

        # decoder: implements how to select the next tokens, given the autoregressive distribution
        if options.beam_size is not None:
            self.decoder = BeamSearchDecoder(
                options.beam_size, tokenizer.eot, self.inference, options.patience
            )
        else:
            self.decoder = GreedyDecoder(options.temperature, tokenizer.eot)

        # logit filters: applies various rules to suppress or penalize certain tokens
        self.logit_filters = []
        if self.options.suppress_blank:
            self.logit_filters.append(SuppressBlank(self.tokenizer, self.sample_begin))
        if self.options.suppress_tokens:
            self.logit_filters.append(SuppressTokens(self._get_suppress_tokens()))

        if not options.without_timestamps:
            precision = CHUNK_LENGTH / model.dims.n_audio_ctx  # usually 0.02 seconds
            max_initial_timestamp_index = None
            if options.max_initial_timestamp:
                max_initial_timestamp_index = round(
                    self.options.max_initial_timestamp / precision
                )
            self.logit_filters.append(
                ApplyTimestampRules(
                    tokenizer, self.sample_begin, max_initial_timestamp_index
                )
            )

    def _verify_options(self, options: DecodingOptions) -> DecodingOptions:
        if options.beam_size is not None and options.best_of is not None:
            raise ValueError("beam_size and best_of can't be given together")
        if options.temperature == 0:
            if options.best_of is not None:
                raise ValueError("best_of with greedy sampling (T=0) is not compatible")
        if options.patience is not None and options.beam_size is None:
            raise ValueError("patience requires beam_size to be given")
        if options.length_penalty is not None and not (
            0 <= options.length_penalty <= 1
        ):
            raise ValueError("length_penalty (alpha) should be a value between 0 and 1")

        return options

    def _get_initial_tokens(self) -> Tuple[int]:
        tokens = list(self.sot_sequence)

        if prefix := self.options.prefix:
            prefix_tokens = (
                self.tokenizer.encode(" " + prefix.strip())
                if isinstance(prefix, str)
                else prefix
            )
            if self.sample_len is not None:
                max_prefix_len = self.n_ctx // 2 - self.sample_len
                prefix_tokens = prefix_tokens[-max_prefix_len:]
            tokens = tokens + prefix_tokens

        if prompt := self.options.prompt:
            prompt_tokens = (
                self.tokenizer.encode(" " + prompt.strip())
                if isinstance(prompt, str)
                else prompt
            )
            tokens = (
                [self.tokenizer.sot_prev]
                + prompt_tokens[-(self.n_ctx // 2 - 1) :]
                + tokens
            )

        return tuple(tokens)

    def _get_suppress_tokens(self) -> Tuple[int]:
        suppress_tokens = self.options.suppress_tokens

        if isinstance(suppress_tokens, str):
            suppress_tokens = [int(t) for t in suppress_tokens.split(",")]

        if -1 in suppress_tokens:
            suppress_tokens = [t for t in suppress_tokens if t >= 0]
            suppress_tokens.extend(self.tokenizer.non_speech_tokens)
        elif suppress_tokens is None or len(suppress_tokens) == 0:
            suppress_tokens = []  # interpret empty string as an empty list
        else:
            assert isinstance(suppress_tokens, list), "suppress_tokens must be a list"

        suppress_tokens.extend(
            [
                self.tokenizer.transcribe,
                self.tokenizer.translate,
                self.tokenizer.sot,
                self.tokenizer.sot_prev,
                self.tokenizer.sot_lm,
            ]
        )
        if self.tokenizer.no_speech is not None:
            # no-speech probability is collected separately
            suppress_tokens.append(self.tokenizer.no_speech)

        return tuple(sorted(set(suppress_tokens)))

    def _get_audio_features(self, mel: np.ndarray):
        mel = mel.astype(np.float16)

        if mel.shape[-2:] == (
            self.model.dims.n_audio_ctx,
            self.model.dims.n_audio_state,
        ):
            # encoded audio features are given; skip audio encoding
            audio_features = mel
        else:
            start_time = time.time()

            mel = mel if mel.flags.c_contiguous else np.ascontiguousarray(mel)
            self.model.encoder_input_tensors_map[self.model.encoder_input_names[0]].update_data(fp16_cast(mel));

            self.model.combined_whisper_engine.process(self.model.encoder_engine_graph_name, self.model.encoder_input_tensors_map, self.model.encoder_output_tensors_map)
            mel_out_tensor = list(self.model.encoder_output_tensors_map.values())[0]

            audio_features = uint16_to_fp16(mel_out_tensor.asnumpy())
            self.model.call_encoder +=1
            self.model.time += time.time() - start_time

        return audio_features

    def _detect_language(self, audio_features: np.ndarray, tokens: np.ndarray):
        languages = [self.options.language] * audio_features.shape[0]
        lang_probs = None

        if self.options.language is None or self.options.task == "lang_id":
            lang_tokens, lang_probs = self.model.detect_language(
                audio_features, self.tokenizer
            )
            languages = [max(probs, key=probs.get) for probs in lang_probs]
            if self.options.language is None:
                tokens[:, self.sot_index + 1] = lang_tokens  # write language tokens

        return languages, lang_probs

    def _main_loop_sail(self, audio_features: np.ndarray, tokens: np.ndarray):
        self.model.main_loop_cnt += 1
        n_batch = tokens.shape[0]
        sum_logprobs = np.zeros(n_batch)
        no_speech_probs = [np.nan] * n_batch
        initial_tokens_length = len(self.initial_tokens)
        padding_num = self.padding_size

        attention_mask_firstly = np.triu(np.full((padding_num, padding_num), -10000), 1)
        attention_mask_with_kvcache_max = np.triu(np.full((448, 448), -10000), 1)
        attention_mask_with_kvcache = attention_mask_with_kvcache_max[-padding_num:, -padding_num:]

        try:
            for i in range(self.sample_len):
                if i == 0:
                    tokens_input = np.pad(tokens, ((0, 0), (padding_num - tokens.shape[1], 0)), mode='constant', constant_values=0)

                    positional_embedding_input = self.model.positional_embedding[:i + initial_tokens_length]

                    padding_width = [(padding_num - initial_tokens_length - i, 0), (0, 0)]

                    positional_embedding_input = np.pad(positional_embedding_input, padding_width, mode='constant', constant_values=0)

                    mask = np.pad(attention_mask_firstly[:tokens.shape[-1], :tokens.shape[-1]], ((0, 0), (padding_num - tokens.shape[-1], 0)), mode='constant', constant_values=-10000)

                    padding_amount = max(padding_num - tokens.shape[-1], 0)
                    pad_width = [(0, 0)] * (mask.ndim - 2) + [(padding_amount, 0), (0, 0)]
                    mask = np.pad(
                        mask,
                        pad_width=pad_width,
                        mode='constant',
                        constant_values=0
                    )

                    mask = mask.reshape(1, 1, *mask.shape)
                    mask = np.repeat(mask, repeats=n_batch, axis=0)
                    mask = np.repeat(mask, repeats=self.n_text_head, axis=1)
                    mask = mask.transpose(0, 2, 1, 3)
                else:
                    tokens_input = tokens[:, -1:]
                    offset = i + initial_tokens_length - 1
                    positional_embedding_input = self.model.positional_embedding[offset:offset+1]
                    mask = attention_mask_with_kvcache[offset:offset+1]
                    mask = np.flip(mask, axis=1)
                    mask = mask.copy()
                    mask = mask.reshape(1, 1, *mask.shape)
                    mask = np.tile(mask, (n_batch, self.n_text_head, 1, 1))
                    mask = mask.transpose(0, 2, 1, 3)
                if i == 0:
                    start_time = time.time()
                    # type transform for decoder_main inputs
                    tokens_input = tokens_input.astype(np.int32)
                    audio_features = audio_features.astype(np.float16)
                    positional_embedding_input = positional_embedding_input.astype(np.float16)
                    mask = mask.astype(np.float16)

                    tokens_input = tokens_input if tokens_input.flags.c_contiguous else np.ascontiguousarray(tokens_input)
                    audio_features = audio_features if audio_features.flags.c_contiguous else np.ascontiguousarray(audio_features)
                    positional_embedding_input = positional_embedding_input if positional_embedding_input.flags.c_contiguous else np.ascontiguousarray(positional_embedding_input)
                    mask = mask if mask.flags.c_contiguous else np.ascontiguousarray(mask)

                    self.model.decoder_main_input_tensors_map[self.model.decoder_main_input_names[0]].update_data(tokens_input)

                    self.model.decoder_main_input_tensors_map[self.model.decoder_main_input_names[1]].update_data(fp16_cast(audio_features))

                    self.model.decoder_main_input_tensors_map[self.model.decoder_main_input_names[2]].update_data(fp16_cast(positional_embedding_input))

                    self.model.decoder_main_input_tensors_map[self.model.decoder_main_input_names[3]].update_data(fp16_cast(mask))

                    self.model.combined_whisper_engine.process(self.model.decoder_main_graph_name, self.model.decoder_main_input_tensors_map,self.model.decoder_main_output_tensors_map)

                    x_tensor = self.model.decoder_main_output_tensors_map[self.model.combined_whisper_engine.get_output_names(self.model.decoder_main_graph_name)[0]]

                    x = uint16_to_fp16(x_tensor.asnumpy())
                    # get input data for decoder_post
                    # this process is dynamic
                    x_sot = x[:, padding_num - initial_tokens_length + self.sot_index:padding_num - initial_tokens_length + self.sot_index + 1].copy()
                    x_last = x[:, -1:].copy()

                    self.model.decoder_post_input_tensors_map[self.model.decoder_post_input_names[0]].update_data(fp16_cast(x_sot));

                    self.model.decoder_post_input_tensors_map[self.model.decoder_post_input_names[1]].update_data(fp16_cast(x_last));

                    self.model.combined_whisper_engine.process(self.model.decoder_post_graph_name, self.model.decoder_post_input_tensors_map, self.model.decoder_post_output_tensors_map)
                    logits_tensor = self.model.decoder_post_output_tensors_map[self.model.decoder_post_output_names[0]]
                    no_speech_probs_tensor = self.model.decoder_post_output_tensors_map[self.model.decoder_post_output_names[1]]

                    logits = uint16_to_fp16(logits_tensor.asnumpy())
                    no_speech_probs = uint16_to_fp16(no_speech_probs_tensor.asnumpy()).tolist()

                    self.model.call_decoder_firstly += 1
                    self.model.time += time.time() - start_time

                else:
                    start_time = time.time()
                    # type transform for decoder_loop inputs
                    tokens_input = tokens_input.astype(np.int32)
                    positional_embedding_input = positional_embedding_input.astype(np.float16)
                    mask = mask.astype(np.float16)

                    tokens_input = tokens_input if tokens_input.flags.contiguous else np.ascontiguousarray(tokens_input)
                    positional_embedding_input = positional_embedding_input if positional_embedding_input.flags.contiguous else np.ascontiguousarray(positional_embedding_input)
                    mask = mask if mask.flags.contiguous else np.ascontiguousarray(mask)

                    # sail
                    self.model.decoder_loop_input_tensors_map[self.model.decoder_loop_input_names[0]].update_data(tokens_input)
                    self.model.decoder_loop_input_tensors_map[self.model.decoder_loop_input_names[1]].update_data(fp16_cast(positional_embedding_input))
                    self.model.decoder_loop_input_tensors_map[self.model.decoder_loop_input_names[1]]

                    self.model.decoder_loop_input_tensors_map[self.model.decoder_loop_input_names[2]].update_data(fp16_cast(mask))
                    self.model.decoder_loop_input_tensors_map[self.model.decoder_loop_input_names[2]]

                    self.model.combined_whisper_engine.process(self.model.decoder_loop_graph_name, self.model.decoder_loop_input_tensors_map, self.model.decoder_loop_output_tensors_map)

                    logits_tensor = self.model.decoder_loop_output_tensors_map[self.model.decoder_loop_output_names[0]]
                    logits = uint16_to_fp16(logits_tensor.asnumpy())

                    self.model.call_decoder_loop += 1
                    self.model.time += time.time() - start_time

                # apply the logit filters, e.g. for suppressing or applying penalty to
                for logit_filter in self.logit_filters:
                    logit_filter.apply(logits, tokens)

                # expand the tokens tensor with the selected next tokens
                tokens, completed = self.decoder.update(tokens,
                                                        logits.astype(np.float32),
                                                        sum_logprobs)

                if completed or tokens.shape[-1] > self.n_ctx:
                    break
        finally:
            pass
        return tokens, sum_logprobs, no_speech_probs

    def run(self, mel: np.ndarray) -> List[DecodingResult]:
        self.decoder.reset()
        tokenizer: Tokenizer = self.tokenizer
        n_audio: int = mel.shape[0]

        audio_features: np.ndarray = self._get_audio_features(mel)  # encoder
        tokens: np.ndarray = np.tile(np.array(self.initial_tokens)[None, :], (n_audio, 1))
        # detect language if requested, overwriting the language token
        languages, language_probs = self._detect_language(audio_features, tokens) # encoder forward pass

        if self.options.task == "lang_id":
            return [
                DecodingResult(
                    audio_features=features, language=language, language_probs=probs
                )
                for features, language, probs in zip(
                    audio_features, languages, language_probs
                )
            ]

        # repeat text tensors by the group size, for beam search or best-of-n sampling

        tokens = np.repeat(tokens, self.n_group, axis=0).astype(np.int32)
        # call the main sampling loop
        tokens, sum_logprobs, no_speech_probs = self._main_loop_sail(audio_features, tokens) # decoder forward pass

        # reshape the tensors to have (n_audio, n_group) as the first two dimensions
        audio_features = audio_features[:: self.n_group]
        no_speech_probs = no_speech_probs[:: self.n_group]
        assert audio_features.shape[0] == len(no_speech_probs) == n_audio

        tokens = tokens.reshape(n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)

        # get the final candidates for each group, and slice between the first sampled token and EOT
        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)
        tokens = [
            [t[self.sample_begin : np.where(t == tokenizer.eot)[0][0]] for t in s]
            for s in tokens
        ]

        # select the top-ranked sample in each group
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens: List[List[int]] = [t[i].tolist() for i, t in zip(selected, tokens)]
        texts: List[str] = [tokenizer.decode(t).strip() for t in tokens]

        sum_logprobs: List[float] = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        avg_logprobs: List[float] = [
            lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)
        ]

        fields = (
            texts,
            languages,
            tokens,
            audio_features,
            avg_logprobs,
            no_speech_probs,
        )
        if len(set(map(len, fields))) != 1:
            raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")

        return [
            DecodingResult(
                audio_features=features,
                language=language,
                tokens=tokens,
                text=text,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                temperature=self.options.temperature,
                compression_ratio=compression_ratio(text),
            )
            for text, language, tokens, features, avg_logprob, no_speech_prob in zip(
                *fields
            )
        ]


def decode(
    model: "Whisper",
    mel: np.ndarray,
    options: DecodingOptions = DecodingOptions(),
    **kwargs,
) -> Union[DecodingResult, List[DecodingResult]]:
    """
    Performs decoding of 30-second audio segment(s), provided as Mel spectrogram(s).

    Parameters
    ----------
    model: Whisper
        the Whisper model instance

    mel: np.ndarray, shape = (80, 3000) or (*, 80, 3000)
        A np.ndarray containing the Mel spectrogram(s)

    options: DecodingOptions
        A dataclass that contains all necessary options for decoding 30-second segments

    Returns
    -------
    result: Union[DecodingResult, List[DecodingResult]]
        The result(s) of decoding contained in `DecodingResult` dataclass instance(s)
    """

    if single := mel.ndim == 2:
        mel = np.expand_dims(mel, 0)

    if kwargs:
        options = replace(options, **kwargs)

    result = DecodingTask(model, options).run(mel)

    return result[0] if single else result
