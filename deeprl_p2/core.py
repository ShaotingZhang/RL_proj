"""Core classes."""
import random
from collections import deque
import numpy as np

class ReplayMemory:
    """Interface for replay memories.

    We have found this to be a useful interface for the replay memory. Feel free to add, modify or delete methods/attributes to this class.

    It is expected that the replay memory has implemented the __iter__, __getitem__, and __len__ methods.


    Methods
    -------
    append(state, action, reward, debug_info=None)
    Add a sample to the replay memory. The sample can be any python object, but it is suggested that tensorflow_rl.core.Sample be used.
    end_episode(final_state, is_terminal, debug_info=None)
    Set the final state of an episode and mark whether it was a true terminal state (i.e. the env returned is_terminal=True), of it is is an artificial terminal state (i.e. agent quit the episode early, but agent could have kept running episode).
    sample(batch_size, indexes=None)
    Return list of samples from the memory.
    Each class will implement a different method of choosing the samples. 
    Optionally, specify the sample indexes manually.
    clear()
    Reset the memory. Deletes all references to the samples.
    """
    def __init__(self, max_size, window_length):
        """Setup memory.

        We should specify the maximum size o the memory.
        Once the memory fills up oldest values should be removed. 
        We can try the collections.deque class as the underlying storage, but our sample method will be very slow.

        We recommend using a list as a ring buffer.
        Just track the index where the next sample should be inserted in the list.
        """
        # the size of picture side
        self.psize = 84
        # the size of circular array
        self.max_size = max_size
        # the length of window which means in one state how many frames we have
        self.window_length = window_length
        # the size of memory
        self.mem_size = (max_size + window_length-1);
        # the memory state
        self.mem_state = np.ones((self.mem_size, self.psize, self.psize), dtype=np.uint8)
        # the memory action
        self.mem_action = np.ones(self.mem_size, dtype=np.int8)
        # the memory reward
        self.mem_reward = np.ones(self.mem_size, dtype=np.float32)
        # the memory terminal
        self.mem_terminal = np.ones(self.mem_size, dtype=np.bool)
        self.start = 0
        # End point to the next position.
        # The content doesn't change when end points at it but change
        # when end points move forward from it.
        self.end = 0
        self.full = False

    # add a state to core
    def append(self, state, action, reward, is_terminal):
        if self.start == 0 and self.end == 0: # the first frame
            # Case 1: 1 2 3 S E. Circular array is empty and we need to repeat the first frame 4 times. One window need 4 frames 
            for i in range(self.window_length-1):
                self.mem_state[i] = state
                self.start = (self.start + 1) % self.mem_size
            self.mem_state[self.start] = state
            self.mem_action[self.start] = action
            self.mem_reward[self.start] = reward
            self.mem_terminal[self.start] = is_terminal
            self.end = (self.start + 1) % self.mem_size
        else:
            # Case 2:  1 2 3 S ... E. Circular array is not empty and not full
            self.mem_state[self.end] = state
            self.mem_action[self.end] = action
            self.mem_reward[self.end] = reward
            self.mem_terminal[self.end] = is_terminal
            self.end = (self.end + 1) % self.mem_size
            if self.end > 0 and self.end < self.start:
                self.full = True
            # Case 3:  ... E 1 2 3 S ... Circular array is full
            if self.full:
                self.start = (self.start + 1) % self.mem_size

    # pick up sample(window) from core
    def sample(self, batch_size, indexes=None):
        # core is empty
        if self.end == 0 and self.start == 0:
            # state, action, reward, next_state, is_terminal
            return None, None, None, None, None
        else:
            count = 0
            if self.end > self.start:
                count = self.end - self.start
            else:
                count = self.max_size

            if count <= batch_size:
                indices = np.arange(0, count-1)
            else:
                #indices range is 0 ... count-2
                indices = np.random.randint(0, count-1, size=batch_size)

            # 4 is the current state frame because of our design. 1 to 4 is current window.
            indices_5 = (self.start + indices + 1) % self.mem_size
            indices_4 = (self.start + indices) % self.mem_size
            indices_3 = (self.start + indices - 1) % self.mem_size
            indices_2 = (self.start + indices - 2) % self.mem_size
            indices_1 = (self.start + indices - 3) % self.mem_size
            frame_5 = self.mem_state[indices_5]
            frame_4 = self.mem_state[indices_4]
            frame_3 = self.mem_state[indices_3]
            frame_2 = self.mem_state[indices_2]
            frame_1 = self.mem_state[indices_1]


            state_list = np.array([frame_1, frame_2, frame_3, frame_4])
            state_list = np.transpose(state_list, [1,0,2,3])

            next_state_list = np.array([frame_2, frame_3, frame_4, frame_5])
            next_state_list = np.transpose(next_state_list, [1,0,2,3])

            action_list = self.mem_action[indices_4]
            reward_list = self.mem_reward[indices_4]
            terminal_list = self.mem_terminal[indices_4]

            return state_list, action_list, reward_list, next_state_list, terminal_list

    # reset the circular array
    def clear(self):
        self.start = 0
        self.end = 0
        self.full = False
