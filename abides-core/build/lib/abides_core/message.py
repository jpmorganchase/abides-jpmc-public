from dataclasses import dataclass, field
from typing import ClassVar, List


@dataclass
class Message:
    """The base Message class no longer holds envelope/header information, however any
    desired information can be placed in the arbitrary body.

    Delivery metadata is now handled outside the message itself.

    The body may be overridden by specific message type subclasses.
    """

    # The autoincrementing variable here will ensure that, when Messages are due for
    # delivery at the same time step, the Message that was created first is delivered
    # first. (Which is not important, but Python 3 requires a fully resolved chain of
    # priority in all cases, so we need something consistent) We might want to generate
    # these with stochasticity, but guarantee uniqueness somehow, to make delivery of
    # orders at the same exact timestamp "random" instead of "arbitrary" (FIFO among
    # tied times) as it currently is.
    __message_id_counter: ClassVar[int] = 1
    message_id: int = field(init=False)

    def __post_init__(self):
        self.message_id: int = Message.__message_id_counter
        Message.__message_id_counter += 1

    def __lt__(self, other: "Message") -> bool:
        # Required by Python3 for this object to be placed in a priority queue.
        return self.message_id < other.message_id

    def type(self) -> str:
        return self.__class__.__name__


@dataclass
class MessageBatch(Message):
    """
    Helper used for batching multiple messages being sent by the same sender to the same
    destination together. If very large numbers of messages are being sent this way,
    using this class can help performance.
    """

    messages: List[Message]


@dataclass
class WakeupMsg(Message):
    """
    Empty message sent to agents when woken up.
    """

    pass
