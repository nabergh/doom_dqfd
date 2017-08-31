# base code from ppaquette_gym_doom.wrappers.action_space

import gym
from ppaquette_gym_doom.wrappers.multi_discrete import BoxToMultiDiscrete, DiscreteToMultiDiscrete

# Constants
NUM_ACTIONS = 43
ALLOWED_ACTIONS = [
    [0, 10, 11],                                # 0 - Basic
    [0, 10, 11, 13, 14, 15],                    # 1 - Corridor
    [0, 14, 15],                                # 2 - DefendCenter
    [0, 14, 15],                                # 3 - DefendLine
    [13, 14, 15],                               # 4 - HealthGathering
    [13, 14, 15],                               # 5 - MyWayHome
    [0, 14, 15],                                # 6 - PredictPosition
    [10, 11],                                   # 7 - TakeCover
    [x for x in range(NUM_ACTIONS) if x != 33], # 8 - Deathmatch
]

__all__ = [ 'ToDiscrete', 'ToBox' ]

def ToDiscrete(config):
    # Config can be 'minimal', 'constant-7', 'constant-17', 'full'

    class ToDiscreteWrapper(gym.Wrapper):
        """
            Doom wrapper to convert MultiDiscrete action space to Discrete
            config:
                - minimal - Will only use the levels' allowed actions (+ NOOP)
                - constant-7 - Will use the 7 minimum actions (+NOOP) to complete all levels
                - constant-17 - Will use the 17 most common actions (+NOOP) to complete all levels
                - full - Will use all available actions (+ NOOP)
            list of commands:
                - minimal:
                    Basic:              NOOP, ATTACK, MOVE_RIGHT, MOVE_LEFT
                    Corridor:           NOOP, ATTACK, MOVE_RIGHT, MOVE_LEFT, MOVE_FORWARD, TURN_RIGHT, TURN_LEFT
                    DefendCenter        NOOP, ATTACK, TURN_RIGHT, TURN_LEFT
                    DefendLine:         NOOP, ATTACK, TURN_RIGHT, TURN_LEFT
                    HealthGathering:    NOOP, MOVE_FORWARD, TURN_RIGHT, TURN_LEFT
                    MyWayHome:          NOOP, MOVE_FORWARD, TURN_RIGHT, TURN_LEFT
                    PredictPosition:    NOOP, ATTACK, TURN_RIGHT, TURN_LEFT
                    TakeCover:          NOOP, MOVE_RIGHT, MOVE_LEFT
                    Deathmatch:         NOOP, ALL COMMANDS (Deltas are limited to [0,1] range and will not work properly)
                - constant-7: NOOP, ATTACK, MOVE_RIGHT, MOVE_LEFT, MOVE_FORWARD, TURN_RIGHT, TURN_LEFT, SELECT_NEXT_WEAPON
                - constant-17: NOOP, ATTACK, JUMP, CROUCH, TURN180, RELOAD, SPEED, STRAFE, MOVE_RIGHT, MOVE_LEFT, MOVE_BACKWARD
                                MOVE_FORWARD, TURN_RIGHT, TURN_LEFT, LOOK_UP, LOOK_DOWN, SELECT_NEXT_WEAPON, SELECT_PREV_WEAPON
        """
        def __init__(self, env):
            super(ToDiscreteWrapper, self).__init__(env)
            if config == 'minimal':
                allowed_actions = ALLOWED_ACTIONS[self.unwrapped.level]
            elif config == 'constant-7':
                allowed_actions = [0, 10, 11, 13, 14, 15, 31]
            elif config == 'constant-17':
                allowed_actions = [0, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 31, 32]
            elif config == 'full':
                allowed_actions = None
            else:
                raise gym.error.Error('Invalid configuration. Valid options are "minimal", "constant-7", "constant-17", "full"')
            self.action_space = DiscreteToMultiDiscrete(self.action_space, allowed_actions)
        def _step(self, action):
            return self.env._step(self.action_space(action))

    return ToDiscreteWrapper


# action_list = [
# 	"ATTACK",
# 	"USE",
# 	"JUMP",
# 	"CROUCH",
# 	"TURN180",
# 	"ALTATTACK",
# 	"RELOAD",
# 	"ZOOM",
# 	"SPEED",
# 	"STRAFE",
# 	"MOVE_RIGHT",
# 	"MOVE_LEFT",
# 	"MOVE_BACKWARD",
# 	"MOVE_FORWARD",
# 	"TURN_RIGHT",
# 	"TURN_LEFT",
# 	"LOOK_UP",
# 	"LOOK_DOWN",
# 	"MOVE_UP",
# 	"MOVE_DOWN",
# 	"LAND",
# 	"SELECT_WEAPON1",
# 	"SELECT_WEAPON2",
# 	"SELECT_WEAPON3",
# 	"SELECT_WEAPON4",
# 	"SELECT_WEAPON5",
# 	"SELECT_WEAPON6",
# 	"SELECT_WEAPON7",
# 	"SELECT_WEAPON8",
# 	"SELECT_WEAPON9",
# 	"SELECT_WEAPON0",
# 	"SELECT_NEXT_WEAPON",
# 	"SELECT_PREV_WEAPON",
# 	"DROP_SELECTED_WEAPON",
# 	"ACTIVATE_SELECTED_ITEM",
# 	"SELECT_NEXT_ITEM",
# 	"SELECT_PREV_ITEM",
# 	"DROP_SELECTED_ITEM",
# 	"LOOK_UP_DOWN_DELTA",
# 	"TURN_LEFT_RIGHT_DELTA",
# 	"MOVE_FORWARD_BACKWARD_DELTA",
# 	"MOVE_LEFT_RIGHT_DELTA",
# 	"MOVE_UP_DOWN_DELTA"
# ]
