import logging
import queue
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from . import NanosecondTime
from .agent import Agent
from .message import Message, MessageBatch, WakeupMsg
from .latency_model import LatencyModel
from .utils import fmt_ts, str_to_ns


logger = logging.getLogger(__name__)


class Kernel:
    """
    ABIDES Kernel

    Arguments:
        agents: List of agents to include in the simulation.
        start_time: Timestamp giving the start time of the simulation.
        stop_time: Timestamp giving the end time of the simulation.
        default_computation_delay: time penalty applied to an agent each time it is
            awakened (wakeup or recvMsg).
        default_latency: latency imposed on each computation, modeled physical latency in systems and avoid infinite loop of events happening at the same exact time (in ns)
        agent_latency: legacy parameter, used when agent_latency_model is not defined
        latency_noise:legacy parameter, used when agent_latency_model is not defined
        agent_latency_model: Model of latency used for the network of agents.
        skip_log: if True, no log saved on disk.
        seed: seed of the simulation.
        log_dir: directory where data is store.
        custom_properties: Different attributes that can be added to the simulation
            (e.g., the oracle).
    """

    def __init__(
        self,
        agents: List[Agent],
        start_time: NanosecondTime = str_to_ns("09:30:00"),
        stop_time: NanosecondTime = str_to_ns("16:00:00"),
        default_computation_delay: int = 1,
        default_latency: float = 1,
        agent_latency: Optional[List[List[float]]] = None,
        latency_noise: List[float] = [1.0],
        agent_latency_model: Optional[LatencyModel] = None,
        skip_log: bool = True,
        seed: Optional[int] = None,
        log_dir: Optional[str] = None,
        custom_properties: Optional[Dict[str, Any]] = None,
        random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        custom_properties = custom_properties or {}

        self.random_state: np.random.RandomState = (
            random_state
            or np.random.RandomState(
                seed=np.random.randint(low=0, high=2 ** 32, dtype="uint64")
            )
        )

        # A single message queue to keep everything organized by increasing
        # delivery timestamp.
        self.messages: queue.PriorityQueue[(int, str, Message)] = queue.PriorityQueue()

        # Timestamp at which the Kernel was created.  Primarily used to
        # create a unique log directory for this run.  Also used to
        # print some elapsed time and messages per second statistics.
        self.kernel_wall_clock_start: datetime = datetime.now()

        self.mean_result_by_agent_type: Dict[str, Any] = {}
        self.agent_count_by_type: Dict[str, int] = {}

        # The Kernel maintains a summary log to which agents can write
        # information that should be centralized for very fast access
        # by separate statistical summary programs.  Detailed event
        # logging should go only to the agent's individual log.  This
        # is for things like "final position value" and such.
        self.summary_log: List[Dict[str, Any]] = []
        # variable to say if has already run at least once or not
        self.has_run = False

        for key, value in custom_properties.items():
            setattr(self, key, value)

        # agents must be a list of agents for the simulation,
        #        based on class agent.Agent
        self.agents: List[Agent] = agents

        # Filter for any ABIDES-Gym agents - does not require dependency on ABIDES-gym.
        self.gym_agents: List[Agent] = list(
            filter(
                lambda agent: "CoreGymAgent"
                in [c.__name__ for c in agent.__class__.__bases__],
                agents,
            )
        )

        # Temporary check until ABIDES-gym supports multiple gym agents
        assert (
            len(self.gym_agents) <= 1
        ), "ABIDES-gym currently only supports using one gym agent"

        logger.debug(f"Detected {len(self.gym_agents)} ABIDES-gym agents")

        # Simulation custom state in a freeform dictionary.  Allows config files
        # that drive multiple simulations, or require the ability to generate
        # special logs after simulation, to obtain needed output without special
        # case code in the Kernel.  Per-agent state should be handled using the
        # provided update_agent_state() method.
        self.custom_state: Dict[str, Any] = {}

        # The kernel start and stop time (first and last timestamp in
        # the simulation, separate from anything like exchange open/close).
        self.start_time: NanosecondTime = start_time
        self.stop_time: NanosecondTime = stop_time

        # This is a NanosecondTime that includes the date.
        self.current_time: NanosecondTime = start_time

        # The global seed, NOT used for anything agent-related.
        self.seed: Optional[int] = seed

        # Should the Kernel skip writing agent logs?
        self.skip_log: bool = skip_log

        # If a log directory was not specified, use the initial wallclock.
        self.log_dir: str = log_dir or str(
            int(self.kernel_wall_clock_start.timestamp())
        )

        # The kernel maintains a current time for each agent to allow
        # simulation of per-agent computation delays.  The agent's time
        # is pushed forward (see below) each time it awakens, and it
        # cannot receive new messages/wakeups until the global time
        # reaches the agent's time.  (i.e. it cannot act again while
        # it is still "in the future")

        # This also nicely enforces agents being unable to act before
        # the simulation start_time.
        self.agent_current_times: List[NanosecondTime] = [self.start_time] * len(
            self.agents
        )

        # agent_computation_delays is in nanoseconds, starts with a default
        # value from config, and can be changed by any agent at any time
        # (for itself only).  It represents the time penalty applied to
        # an agent each time it is awakened  (wakeup or recvMsg).  The
        # penalty applies _after_ the agent acts, before it may act again.
        self.agent_computation_delays: List[int] = [default_computation_delay] * len(
            self.agents
        )

        # If an agent_latency_model is defined, it will be used instead of
        # the older, non-model-based attributes.
        self.agent_latency_model = agent_latency_model

        # If an agent_latency_model is NOT defined, the older parameters:
        # agent_latency (or default_latency) and latency_noise should be specified.
        # These should be considered deprecated and will be removed in the future.

        # If agent_latency is not defined, define it using the default_latency.
        # This matrix defines the communication delay between every pair of
        # agents.
        if agent_latency is None:
            self.agent_latency: List[List[float]] = [
                [default_latency] * len(self.agents)
            ] * len(self.agents)
        else:
            self.agent_latency = agent_latency

        # There is a noise model for latency, intended to be a one-sided
        # distribution with the peak at zero.  By default there is no noise
        # (100% chance to add zero ns extra delay).  Format is a list with
        # list index = ns extra delay, value = probability of this delay.
        self.latency_noise: List[float] = latency_noise

        # The kernel maintains an accumulating additional delay parameter
        # for the current agent.  This is applied to each message sent
        # and upon return from wakeup/receive_message, in addition to the
        # agent's standard computation delay.  However, it never carries
        # over to future wakeup/receive_message calls.  It is useful for
        # staggering of sent messages.
        self.current_agent_additional_delay: int = 0

        self.show_trace_messages: bool = False

        logger.debug(f"Kernel initialized")

    def run(self) -> Dict[str, Any]:
        """
        Wrapper to run the entire simulation (when not running in ABIDES-Gym mode).

        3 Steps:
          - Simulation Instantiation
          - Simulation Run
          - Simulation Termination

        Returns:
            An object that contains all the objects at the end of the simulation.
        """
        self.initialize()

        self.runner()

        return self.terminate()

    # This is called to actually start the simulation, once all agent
    # configuration is done.
    def initialize(self) -> None:
        """
        Instantiation of the simulation:
          - Creation of the different object of the simulation.
          - Instantiation of the latency network
          - Calls on the kernel_initializing and KernelStarting of the different agents
        """

        logger.debug("Kernel started")
        logger.debug("Simulation started!")

        # Note that num_simulations has not yet been really used or tested
        # for anything.  Instead we have been running multiple simulations
        # with coarse parallelization from a shell script

        # Event notification for kernel init (agents should not try to
        # communicate with other agents, as order is unknown).  Agents
        # should initialize any internal resources that may be needed
        # to communicate with other agents during agent.kernel_starting().
        # Kernel passes self-reference for agents to retain, so they can
        # communicate with the kernel in the future (as it does not have
        # an agentID).
        logger.debug("--- Agent.kernel_initializing() ---")
        for agent in self.agents:
            agent.kernel_initializing(self)

        # Event notification for kernel start (agents may set up
        # communications or references to other agents, as all agents
        # are guaranteed to exist now).  Agents should obtain references
        # to other agents they require for proper operation (exchanges,
        # brokers, subscription services...).  Note that we generally
        # don't (and shouldn't) permit agents to get direct references
        # to other agents (like the exchange) as they could then bypass
        # the Kernel, and therefore simulation "physics" to send messages
        # directly and instantly or to perform disallowed direct inspection
        # of the other agent's state.  Agents should instead obtain the
        # agent ID of other agents, and communicate with them only via
        # the Kernel.  Direct references to utility objects that are not
        # agents are acceptable (e.g. oracles).
        logger.debug("--- Agent.kernel_starting() ---")
        for agent in self.agents:
            agent.kernel_starting(self.start_time)

        # Set the kernel to its start_time.
        self.current_time = self.start_time

        logger.debug("--- Kernel Clock started ---")
        logger.debug("Kernel.current_time is now {}".format(fmt_ts(self.current_time)))

        # Start processing the Event Queue.
        logger.debug("--- Kernel Event Queue begins ---")
        logger.debug(
            "Kernel will start processing messages. Queue length: {}".format(
                len(self.messages.queue)
            )
        )

        # Track starting wall clock time and total message count for stats at the end.
        self.event_queue_wall_clock_start = datetime.now()
        self.ttl_messages = 0

    def runner(
        self, agent_actions: Optional[Tuple[Agent, List[Dict[str, Any]]]] = None
    ) -> Dict[str, Any]:
        """
        Start the simulation and processing of the message queue.
        Possibility to add the optional argument agent_actions. It is a list of dictionaries corresponding
        to actions to be performed by the experimental agent (Gym Agent).

        Arguments:
            agent_actions: A list of the different actions to be performed represented in a dictionary per action.

        Returns:
          - it is a dictionnary composed of two elements:
            - "done": boolean True if the simulation is done, else False. It is true when simulation reaches end_time or when the message queue is empty.
            - "results": it is the raw_state returned by the gym experimental agent, contains data that will be formated in the gym environement to formulate state, reward, info etc.. If
               there is no gym experimental agent, then it is None.
        """
        # run an action on a given agent before resuming queue: to be used to take exp agent action before resuming run
        if agent_actions is not None:
            exp_agent, action_list = agent_actions
            exp_agent.apply_actions(action_list)

        # Process messages until there aren't any (at which point there never can
        # be again, because agents only "wake" in response to messages), or until
        # the kernel stop time is reached.
        while (
            not self.messages.empty()
            and self.current_time
            and (self.current_time <= self.stop_time)
        ):
            # Get the next message in timestamp order (delivery time) and extract it.
            self.current_time, event = self.messages.get()
            assert self.current_time is not None

            sender_id, recipient_id, message = event

            # Periodically print the simulation time and total messages, even if muted.
            if self.ttl_messages % 100000 == 0:
                logger.info(
                    "--- Simulation time: {}, messages processed: {:,}, wallclock elapsed: {:.2f}s ---".format(
                        fmt_ts(self.current_time),
                        self.ttl_messages,
                        (
                            datetime.now() - self.event_queue_wall_clock_start
                        ).total_seconds(),
                    )
                )

            if self.show_trace_messages:
                logger.debug("--- Kernel Event Queue pop ---")
                logger.debug(
                    "Kernel handling {} message for agent {} at time {}".format(
                        message.type(), recipient_id, self.current_time
                    )
                )

            self.ttl_messages += 1

            # In between messages, always reset the current_agent_additional_delay.
            self.current_agent_additional_delay = 0

            # Dispatch message to agent.
            if isinstance(message, WakeupMsg):
                # Test to see if the agent is already in the future.  If so,
                # delay the wakeup until the agent can act again.
                if self.agent_current_times[recipient_id] > self.current_time:
                    # Push the wakeup call back into the PQ with a new time.
                    self.messages.put(
                        (
                            self.agent_current_times[recipient_id],
                            (sender_id, recipient_id, message),
                        )
                    )
                    if self.show_trace_messages:
                        logger.debug(
                            "After wakeup return, agent {} delayed from {} to {}".format(
                                recipient_id,
                                fmt_ts(self.current_time),
                                fmt_ts(self.agent_current_times[recipient_id]),
                            )
                        )
                    continue

                # Set agent's current time to global current time for start
                # of processing.
                self.agent_current_times[recipient_id] = self.current_time

                # Wake the agent and get value passed to kernel to listen for kernel interruption signal
                wakeup_result = self.agents[recipient_id].wakeup(self.current_time)

                # Delay the agent by its computation delay plus any transient additional delay requested.
                self.agent_current_times[recipient_id] += (
                    self.agent_computation_delays[recipient_id]
                    + self.current_agent_additional_delay
                )

                if self.show_trace_messages:
                    logger.debug(
                        "After wakeup return, agent {} delayed from {} to {}".format(
                            recipient_id,
                            fmt_ts(self.current_time),
                            fmt_ts(self.agent_current_times[recipient_id]),
                        )
                    )
                # catch kernel interruption signal and return wakeup_result which is the raw state from gym agent
                if wakeup_result != None:
                    return {"done": False, "result": wakeup_result}
            else:
                # Test to see if the agent is already in the future.  If so,
                # delay the message until the agent can act again.
                if self.agent_current_times[recipient_id] > self.current_time:
                    # Push the message back into the PQ with a new time.
                    self.messages.put(
                        (
                            self.agent_current_times[recipient_id],
                            (sender_id, recipient_id, message),
                        )
                    )
                    if self.show_trace_messages:
                        logger.debug(
                            "Agent in future: message requeued for {}".format(
                                fmt_ts(self.agent_current_times[recipient_id])
                            )
                        )
                    continue

                # Set agent's current time to global current time for start
                # of processing.
                self.agent_current_times[recipient_id] = self.current_time

                # Deliver the message.
                if isinstance(message, MessageBatch):
                    messages = message.messages
                else:
                    messages = [message]

                for message in messages:
                    # Delay the agent by its computation delay plus any transient additional delay requested.
                    self.agent_current_times[recipient_id] += (
                        self.agent_computation_delays[recipient_id]
                        + self.current_agent_additional_delay
                    )

                    if self.show_trace_messages:
                        logger.debug(
                            "After receive_message return, agent {} delayed from {} to {}".format(
                                recipient_id,
                                fmt_ts(self.current_time),
                                fmt_ts(self.agent_current_times[recipient_id]),
                            )
                        )

                    self.agents[recipient_id].receive_message(
                        self.current_time, sender_id, message
                    )

        if self.messages.empty():
            logger.debug("--- Kernel Event Queue empty ---")

        if self.current_time and (self.current_time > self.stop_time):
            logger.debug("--- Kernel Stop Time surpassed ---")

        # if gets here means sim queue is fully processed, return to show sim is done
        if len(self.gym_agents) > 0:
            self.gym_agents[0].update_raw_state()
            return {"done": True, "result": self.gym_agents[0].get_raw_state()}
        else:
            return {"done": True, "result": None}

    def terminate(self) -> Dict[str, Any]:
        """
        Termination of the simulation. Called once the queue is empty, or the gym environement is done, or the simulation
        reached kernel stop time:
          - Calls the kernel_stopping of the agents
          - Calls the kernel_terminating of the agents

        Returns:
            custom_state: it is an object that contains everything in the simulation. In particular it is useful to retrieve agents and/or logs after the simulation to proceed to analysis.
        """
        # Record wall clock stop time and elapsed time for stats at the end.
        event_queue_wall_clock_stop = datetime.now()

        event_queue_wall_clock_elapsed = (
            event_queue_wall_clock_stop - self.event_queue_wall_clock_start
        )

        # Event notification for kernel end (agents may communicate with
        # other agents, as all agents are still guaranteed to exist).
        # Agents should not destroy resources they may need to respond
        # to final communications from other agents.
        logger.debug("--- Agent.kernel_stopping() ---")
        for agent in self.agents:
            agent.kernel_stopping()

        # Event notification for kernel termination (agents should not
        # attempt communication with other agents, as order of termination
        # is unknown).  Agents should clean up all used resources as the
        # simulation program may not actually terminate if num_simulations > 1.
        logger.debug("\n--- Agent.kernel_terminating() ---")
        for agent in self.agents:
            agent.kernel_terminating()

        logger.info(
            "Event Queue elapsed: {}, messages: {:,}, messages per second: {:0.1f}".format(
                event_queue_wall_clock_elapsed,
                self.ttl_messages,
                self.ttl_messages / event_queue_wall_clock_elapsed.total_seconds(),
            )
        )

        # The Kernel adds a handful of custom state results for all simulations,
        # which configurations may use, print, log, or discard.
        self.custom_state[
            "kernel_event_queue_elapsed_wallclock"
        ] = event_queue_wall_clock_elapsed
        self.custom_state["kernel_slowest_agent_finish_time"] = max(
            self.agent_current_times
        )
        self.custom_state["agents"] = self.agents

        # Agents will request the Kernel to serialize their agent logs, usually
        # during kernel_terminating, but the Kernel must write out the summary
        # log itself.
        self.write_summary_log()

        # This should perhaps be elsewhere, as it is explicitly financial, but it
        # is convenient to have a quick summary of the results for now.
        logger.info("Mean ending value by agent type:")

        for a in self.mean_result_by_agent_type:
            value = self.mean_result_by_agent_type[a]
            count = self.agent_count_by_type[a]
            logger.info(f"{a}: {int(round(value / count)):d}")

        logger.info("Simulation ending!")

        return self.custom_state

    def reset(self) -> None:
        """
        Used in the gym core environment:
          - First calls termination of the kernel, to close previous simulation
          - Then initializes a new simulation
          - Then runs the simulation (not specifying any action this time).
        """

        if self.has_run:  # meaning at leat initialization has been run once

            self.terminate()

        self.initialize()
        self.runner()

    def send_message(
        self, sender_id: int, recipient_id: int, message: Message, delay: int = 0
    ) -> None:
        """
        Called by an agent to send a message to another agent.

        The kernel supplies its own current_time (i.e. "now") to prevent possible abuse
        by agents. The kernel will handle computational delay penalties and/or network
        latency.

        Arguments:
            sender_id: ID of the agent sending the message.
            recipient_id: ID of the agent receiving the message.
            message: The ``Message`` class instance to send.
            delay: Represents an agent's request for ADDITIONAL delay (beyond the
                Kernel's mandatory computation + latency delays). Represents parallel
                pipeline processing delays (that should delay the transmission of
                messages but do not make the agent "busy" and unable to respond to new
                messages)
        """

        # Apply the agent's current computation delay to effectively "send" the message
        # at the END of the agent's current computation period when it is done "thinking".
        # NOTE: sending multiple messages on a single wake will transmit all at the same
        # time, at the end of computation.  To avoid this, use Agent.delay() to accumulate
        # a temporary delay (current cycle only) that will also stagger messages.

        # The optional pipeline delay parameter DOES push the send time forward, since it
        # represents "thinking" time before the message would be sent.  We don't use this
        # for much yet, but it could be important later.

        # This means message delay (before latency) is the agent's standard computation
        # delay PLUS any accumulated delay for this wake cycle PLUS any one-time
        # requested delay for this specific message only.
        sent_time = (
            self.current_time
            + self.agent_computation_delays[sender_id]
            + self.current_agent_additional_delay
            + delay
        )

        # Apply communication delay per the agent_latency_model, if defined, or the
        # agent_latency matrix [sender_id][recipient_id] otherwise.
        if self.agent_latency_model is not None:
            latency: float = self.agent_latency_model.get_latency(
                sender_id=sender_id, recipient_id=recipient_id
            )
            deliver_at = sent_time + int(latency)
            if self.show_trace_messages:
                logger.debug(
                    "Kernel applied latency {}, accumulated delay {}, one-time delay {} on send_message from: {} to {}, scheduled for {}".format(
                        latency,
                        self.current_agent_additional_delay,
                        delay,
                        self.agents[sender_id].name,
                        self.agents[recipient_id].name,
                        fmt_ts(deliver_at),
                    )
                )
        else:
            latency = self.agent_latency[sender_id][recipient_id]
            noise = self.random_state.choice(
                len(self.latency_noise), p=self.latency_noise
            )
            deliver_at = sent_time + int(latency + noise)
            if self.show_trace_messages:
                logger.debug(
                    "Kernel applied latency {}, noise {}, accumulated delay {}, one-time delay {} on send_message from: {} to {}, scheduled for {}".format(
                        latency,
                        noise,
                        self.current_agent_additional_delay,
                        delay,
                        self.agents[sender_id].name,
                        self.agents[recipient_id].name,
                        fmt_ts(deliver_at),
                    )
                )

        # Finally drop the message in the queue with priority == delivery time.
        self.messages.put((deliver_at, (sender_id, recipient_id, message)))

        if self.show_trace_messages:
            logger.debug(
                "Sent time: {}, current time {}, computation delay {}".format(
                    sent_time,
                    fmt_ts(self.current_time),
                    self.agent_computation_delays[sender_id],
                )
            )
            logger.debug("Message queued: {}".format(message))

    def set_wakeup(
        self, sender_id: int, requested_time: Optional[NanosecondTime] = None
    ) -> None:
        """
        Called by an agent to receive a "wakeup call" from the kernel at some requested
        future time.

        NOTE: The agent is responsible for maintaining any required state; the kernel
        will not supply any parameters to the ``wakeup()`` call.

        Arguments:
            sender_id: The ID of the agent making the call.
            requested_time: Defaults to the next possible timestamp.  Wakeup time cannot
            be the current time or a past time.
        """

        if requested_time is None:
            requested_time = self.current_time + 1

        if self.current_time and (requested_time < self.current_time):
            raise ValueError(
                "set_wakeup() called with requested time not in future",
                "current_time:",
                self.current_time,
                "requested_time:",
                requested_time,
            )

        if self.show_trace_messages:
            logger.debug(
                "Kernel adding wakeup for agent {} at time {}".format(
                    sender_id, fmt_ts(requested_time)
                )
            )

        self.messages.put((requested_time, (sender_id, sender_id, WakeupMsg())))

    def get_agent_compute_delay(self, sender_id: int) -> int:
        """
        Allows an agent to query its current computation delay.

        Arguments:
            sender_id: The ID of the agent to get the computational delay for.
        """
        return self.agent_computation_delays[sender_id]

    def set_agent_compute_delay(self, sender_id: int, requested_delay: int) -> None:
        """
        Called by an agent to update its computation delay.

        This does not initiate a global delay, nor an immediate delay for the agent.
        Rather it sets the new default delay for the calling agent. The delay will be
        applied upon every return from wakeup or recvMsg. Note that this delay IS
        applied to any messages sent by the agent during the current wake cycle
        (simulating the messages popping out at the end of its "thinking" time).

        Also note that we DO permit a computation delay of zero, but this should really
        only be used for special or massively parallel agents.

        Arguments:
            sender_id: The ID of the agent making the call.
            requested_delay: delay given in nanoseconds.
        """

        # requested_delay should be in whole nanoseconds.
        if not isinstance(requested_delay, int):
            raise ValueError(
                "Requested computation delay must be whole nanoseconds.",
                "requested_delay:",
                requested_delay,
            )

        # requested_delay must be non-negative.
        if requested_delay < 0:
            raise ValueError(
                "Requested computation delay must be non-negative nanoseconds.",
                "requested_delay:",
                requested_delay,
            )

        self.agent_computation_delays[sender_id] = requested_delay

    def delay_agent(self, sender_id: int, additional_delay: int) -> None:
        """
        Called by an agent to accumulate temporary delay for the current wake cycle.

        This will apply the total delay (at time of send_message) to each message, and
        will modify the agent's next available time slot.  These happen on top of the
        agent's compute delay BUT DO NOT ALTER IT. (i.e. effects are transient). Mostly
        useful for staggering outbound messages.

        Arguments:
            sender_id: The ID of the agent making the call.
            additional_delay: additional delay given in nanoseconds.
        """

        # additional_delay should be in whole nanoseconds.
        if not isinstance(additional_delay, int):
            raise ValueError(
                "Additional delay must be whole nanoseconds.",
                "additional_delay:",
                additional_delay,
            )

        # additional_delay must be non-negative.
        if additional_delay < 0:
            raise ValueError(
                "Additional delay must be non-negative nanoseconds.",
                "additional_delay:",
                additional_delay,
            )

        self.current_agent_additional_delay += additional_delay

    def find_agents_by_type(self, agent_type: Type[Agent]) -> List[int]:
        """
        Returns the IDs of any agents that are of the given type.

        Arguments:
            type: The agent type to search for.

        Returns:
            A list of agent IDs that are instances of the type.
        """
        return [agent.id for agent in self.agents if isinstance(agent, agent_type)]

    def write_log(
        self, sender_id: int, df_log: pd.DataFrame, filename: Optional[str] = None
    ) -> None:
        """
        Called by any agent, usually at the very end of the simulation just before
        kernel shutdown, to write to disk any log dataframe it has been accumulating
        during simulation.

        The format can be decided by the agent, although changes will require a special
        tool to read and parse the logs.  The Kernel places the log in a unique
        directory per run, with one filename per agent, also decided by the Kernel using
        agent type, id, etc.

        If there are too many agents, placing all these files in a directory might be
        unfortunate. Also if there are too many agents, or if the logs are too large,
        memory could become an issue. In this case, we might have to take a speed hit to
        write logs incrementally.

        If filename is not None, it will be used as the filename. Otherwise, the Kernel
        will construct a filename based on the name of the Agent requesting log archival.

        Arguments:
            sender_id: The ID of the agent making the call.
            df_log: dataframe representation of the log that contains all the events logged during the simulation.
            filename: Location on disk to write the log to.
        """

        if self.skip_log:
            return

        path = os.path.join(".", "log", self.log_dir)

        if filename:
            file = "{}.bz2".format(filename)
        else:
            file = "{}.bz2".format(self.agents[sender_id].name.replace(" ", ""))

        if not os.path.exists(path):
            os.makedirs(path)

        df_log.to_pickle(os.path.join(path, file), compression="bz2")

    def append_summary_log(self, sender_id: int, event_type: str, event: Any) -> None:
        """
        We don't even include a timestamp, because this log is for one-time-only summary
        reporting, like starting cash, or ending cash.

        Arguments:
            sender_id: The ID of the agent making the call.
            event_type: The type of the event.
            event: The event to append to the log.
        """
        self.summary_log.append(
            {
                "AgentID": sender_id,
                "AgentStrategy": self.agents[sender_id].type,
                "EventType": event_type,
                "Event": event,
            }
        )

    def write_summary_log(self) -> None:
        path = os.path.join(".", "log", self.log_dir)
        file = "summary_log.bz2"

        if not os.path.exists(path):
            os.makedirs(path)

        df_log = pd.DataFrame(self.summary_log)

        df_log.to_pickle(os.path.join(path, file), compression="bz2")

    def update_agent_state(self, agent_id: int, state: Any) -> None:
        """
        Called by an agent that wishes to replace its custom state in the dictionary the
        Kernel will return at the end of simulation. Shared state must be set directly,
        and agents should coordinate that non-destructively.

        Note that it is never necessary to use this kernel state dictionary for an agent
        to remember information about itself, only to report it back to the config file.

        Arguments:
            agent_id: The agent to update state for.
            state: The new state.
        """

        if "agent_state" not in self.custom_state:
            self.custom_state["agent_state"] = {}

        self.custom_state["agent_state"][agent_id] = state
