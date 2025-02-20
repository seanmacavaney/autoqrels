__version__ = '0.0.1'

from autoqrels.labeler import Labeler, DummyLabeler
from autoqrels import text
from autoqrels import oneshot

__all__ = ['Labeler', 'DummyLabeler', 'text', 'oneshot']
