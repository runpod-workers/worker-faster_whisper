INPUT_VALIDATIONS = {
    'audio': {
        'type': str|None,
        'required': False,
        'default': None
    },
    'audio_base64': {
        'type': str|None,
        'required': False,
        'default': None
    },
    'model': {
        'type': str,
        'required': False,
        'default': 'large-v3'
    },
    'transcription': {
        'type': str,
        'required': False,
        'default': 'plain_text'
    },
    'multilingual': {
        'type': bool,
        'required': False,
        'default': False
    },
    'translate': {
        'type': bool,
        'required': False,
        'default': False
    },
    'translation': {
        'type': str,
        'required': False,
        'default': 'plain_text'
    },
    'language': {
        'type': str|None,
        'required': False,
        'default': None
    },
    'temperature': {
        'type': float,
        'required': False,
        'default': 0.0
    },
    'best_of': {
        'type': int,
        'required': False,
        'default': 5
    },
    'beam_size': {
        'type': int,
        'required': False,
        'default': 5
    },
    'patience': {
        'type': float,
        'required': False,
        'default': 1.0
    },
    'length_penalty': {
        'type': float,
        'required': False,
        'default': 0.0
    },
    'suppress_tokens': {
        'type': str,
        'required': False,
        'default': '-1'
    },
    'initial_prompt': {
        'type': str|None,
        'required': False,
        'default': None
    },
    'condition_on_previous_text': {
        'type': bool,
        'required': False,
        'default': True
    },
    'temperature_increment_on_fallback': {
        'type': float,
        'required': False,
        'default': 0.2
    },
    'compression_ratio_threshold': {
        'type': float,
        'required': False,
        'default': 2.4
    },
    'logprob_threshold': {
        'type': float,
        'required': False,
        'default': -1.0
    },
    'no_speech_threshold': {
        'type': float,
        'required': False,
        'default': 0.6
    },
    'enable_vad': {
        'type': bool,
        'required': False,
        'default': True
    },
    'word_timestamps': {
        'type': bool,
        'required': False,
        'default': False
    },
    'batch_size': {
        'type': int,
        'required': False,
        'default': 8
    }
}
