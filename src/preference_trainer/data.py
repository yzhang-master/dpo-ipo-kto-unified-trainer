from dataclasses import dataclass


@dataclass(frozen=True)
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str


_TRAIN_ROWS = [
    ("Capital of France?", " Paris.", " London."),
    ("Capital of Germany?", " Berlin.", " Paris."),
    ("Capital of Italy?", " Rome.", " Madrid."),
    ("Capital of Spain?", " Madrid.", " Rome."),
    ("Two plus two?", " Four.", " Five."),
    ("Three plus three?", " Six.", " Seven."),
    ("Five minus two?", " Three.", " Two."),
    ("Ten minus four?", " Six.", " Eight."),
    ("Color of clear daytime sky?", " Blue.", " Green."),
    ("Color of fresh grass?", " Green.", " Blue."),
    ("Water freezes at zero Celsius?", " Yes.", " No."),
    ("The Sun rises in the east?", " Yes.", " No."),
    ("A triangle has three sides?", " Yes.", " No."),
    ("A square has five sides?", " No.", " Yes."),
    ("Opposite of hot?", " Cold.", " Warm."),
    ("Opposite of dark?", " Light.", " Black."),
    ("A baby dog is a?", " Puppy.", " Kitten."),
    ("A baby cat is a?", " Kitten.", " Puppy."),
    ("Birds use wings to?", " Fly.", " Swim."),
    ("Fish use fins to?", " Swim.", " Fly."),
    ("First letter of alphabet?", " A.", " Z."),
    ("Last letter of alphabet?", " Z.", " A."),
    ("One dozen equals?", " Twelve.", " Ten."),
    ("One week has seven days?", " Yes.", " No."),
]

_EVAL_ROWS = [
    ("Capital of Japan?", " Tokyo.", " Paris."),
    ("Capital of Canada?", " Ottawa.", " London."),
    ("Four plus four?", " Eight.", " Six."),
    ("Six minus one?", " Five.", " Four."),
    ("Color of ripe lemons?", " Yellow.", " Blue."),
    ("Ice is solid water?", " Yes.", " No."),
    ("Opposite of up?", " Down.", " Left."),
    ("A baby cow is a?", " Calf.", " Puppy."),
]


def _format(rows: list[tuple[str, str, str]]) -> list[PreferencePair]:
    return [
        PreferencePair(f"Q: {question}\nA:", chosen, rejected)
        for question, chosen, rejected in rows
    ]


def preference_datasets() -> tuple[list[PreferencePair], list[PreferencePair]]:
    """Return fixed, disjoint train and evaluation preference pairs."""
    return _format(_TRAIN_ROWS), _format(_EVAL_ROWS)
