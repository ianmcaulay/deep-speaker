import logging
from pathlib import Path
import sys
from natsort import natsorted
from glob import glob
import librosa
import numpy as np
from scipy.spatial.distance import cosine

from train_cli import triplet_softmax_model
from audio_reader import AudioReader, trim_silence
from constants import c
from speech_features import get_mfcc_features_390
from utils import normalize


logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VCTK_PATH = Path('/Users/ianmcaulay/Programming/stargans/StarGAN-Voice-Conversion-master/data/VCTK-Corpus/')
assert VCTK_PATH.exists()
#input_audio_dir = VCTK_PATH.joinpath('wav16')
input_audio_dir = Path('samples/PhilippeRemy/')
assert input_audio_dir.exists()
cache_dir = Path('cache')


def load_model():
    m = triplet_softmax_model(num_speakers_softmax=len(c.AUDIO.SPEAKERS_TRAINING_SET),
                              emb_trainable=False,
                              normalize_embeddings=True,
                              batch_size=None)

    checkpoints = natsorted(glob('checkpoints/*.h5'))
    print(m.summary())

    if len(checkpoints) != 0:
        checkpoint_file = checkpoints[-1]
        initial_epoch = int(checkpoint_file.split('/')[-1].split('.')[0].split('_')[-1])
        logger.info('Initial epoch is {}.'.format(initial_epoch))
        logger.info('Loading checkpoint: {}.'.format(checkpoint_file))
        m.load_weights(checkpoint_file)  # latest one.
    return m


model = load_model()


def wavs_to_vec(wavs, iters=5000):
    #voice_audios = sorted(glob(wav_dir + '/*.wav'))[:max_files]
    wavs = sorted(wavs)
    voice_audios = [get_voice_from_file(wav) for wav in wavs]
    features = []
    i = 0
    #while i % len(voice_audios) < len(v)
    #for i, voice_audio in enumerate(voice_audios):
    while i < iters:
        voice_audio = voice_audios[i % len(voice_audios)]

        cuts = np.random.uniform(low=1, high=len(voice_audio), size=2)
        signal_to_process = voice_audio[int(min(cuts)):int(max(cuts))]
        features_for_single = get_mfcc_features_390(signal_to_process, c.AUDIO.SAMPLE_RATE, max_frames=None)
        # if len(features_per_conv) > 0:
        #     features.append(features_per_conv)

        if len(features_for_single) == 0:
            print(f'0 length features for {wavs[i % len(voice_audios)]}')
        #     import pdb
        #     pdb.set_trace()
        # assert len(features_for_single) > 0
        else:
            features.append(features_for_single)

        i += 1

    #speaker_cache, metadata = self.audio_reader.load_cache([speaker_id])
    #audio_entities = list(speaker_cache.values())
    #logger.info('Generating the inputs necessary for the inference (speaker is {})...'.format(speaker_id))
    #logger.info('This might take a couple of minutes to complete.')
    #feat = generate_features(audio_entities, self.max_count_per_class, progress_bar=False)
    # mean = np.mean([np.mean(t) for t in feat])
    # std = np.mean([np.std(t) for t in feat])
    # feat = normalize(feat, mean, std)

    mean = np.mean([np.mean(t) for t in features])
    std = np.mean([np.std(t) for t in features])
    features = normalize(features, mean, std)

    stacked_embeddings = model.predict(np.vstack(features))[0]
    #emb_sp2 = m.predict(np.vstack(sp2_feat))[0]

    logger.info('Checking that L2 norm is 1.')
    logger.info(np.mean(np.linalg.norm(stacked_embeddings, axis=1)))

    embeddings = stacked_embeddings.mean(axis=0)
    return embeddings
    # import pdb
    # pdb.set_trace()
    #
    # embed = model.predict(features_per_conv)[0]
    # # TODO(ian): need to normalize here?
    # embed = np.mean(embed, axis=0)
    # return embed

    #return model.predict(features_per_conv)[0]


def get_voice_from_file(filename):
    audio, _ = librosa.load(filename, sr=SAMPLE_RATE, mono=True)
    audio = audio.reshape(-1, 1)
    energy = np.abs(audio[:, 0])
    silence_threshold = np.percentile(energy, 95)
    offsets = np.where(energy > silence_threshold)[0]
    # left_blank_duration_ms = (1000.0 * offsets[0]) // sample_rate  # frame_id to duration (ms)
    # right_blank_duration_ms = (1000.0 * (len(audio) - offsets[-1])) // sample_rate
    audio_voice_only = audio[offsets[0]:offsets[-1]]
    return audio_voice_only


#SAMPLE_RATE = 16000
SAMPLE_RATE = c.AUDIO.SAMPLE_RATE
# wav_to_vec('samples/PhilippeRemy/PhilippeRemy_001.wav')

philip_wavs = sorted(glob('samples/PhilippeRemy/*.wav'))
p225_wavs = sorted(glob(str(VCTK_PATH.joinpath('wav16', 'p225', '*.wav'))))
p226_wavs = sorted(glob(str(VCTK_PATH.joinpath('wav16', 'p226', '*.wav'))))

#philip_embeds = [wav_to_vec(wav) for wav in philip_wavs]
#p225_embeds = [wav_to_vec(wav) for wav in p225_wavs]
# philip_embed = wavs_to_vec(philip_wavs)
# philip_embed2 = wavs_to_vec(philip_wavs)
# p225_embed = wavs_to_vec(p225_wavs[:40])
# p225_embed_last = wavs_to_vec(p225_wavs[-40:])
# p226_embed = wavs_to_vec(p226_wavs)
# p226_embed2 = wavs_to_vec(p226_wavs[:50])

d = {
    'philip_embed': wavs_to_vec(philip_wavs),
    'philip_embed2': wavs_to_vec(philip_wavs),
    'p225_embed': wavs_to_vec(p225_wavs[:40]),
    'p225_embed_last': wavs_to_vec(p225_wavs[-40:]),
    'p226_embed': wavs_to_vec(p226_wavs),
    'p226_embed2': wavs_to_vec(p226_wavs[:50]),
}

keys = list(d.keys())
for i in range(len(keys) - 1):
    key1 = keys[i]
    for key2 in keys[i+1:]:
        print(f'{key1} vs {key2} = {cosine(d[key1], d[key2])}')

# print(f'Philip vs Philip2 = {cosine(philip_embed, philip_embed2)}')
# print(f'Philip vs p225 = {cosine(philip_embed, p225_embed)}')
# print(f'Philip vs p225 last = {cosine(philip_embed, p225_embed_last)}')
# print(f'p225 first four vs p225 last four = {cosine(p225_embed, p225_embed_last)}')

import pdb
pdb.set_trace()

audio_reader = AudioReader(input_audio_dir=input_audio_dir,
                           output_cache_dir=cache_dir,
                           sample_rate=c.AUDIO.SAMPLE_RATE,
                           multi_threading=True)
#audio_reader.build_cache()
# print(audio_reader.all_speaker_ids)
# import pdb
# pdb.set_trace()
#regenerate_full_cache(audio_reader, cache_dir)

#unseen_speakers = ['p225', 'PhilippeRemy']

#inference_unseen_speakers(audio_reader, 'p225', 'PhilippeRemy')

#speaker_id = 'p225'
#from unseen_speakers import inference_embeddings
#inference_embeddings(audio_reader, 'PhilippeRemy')




