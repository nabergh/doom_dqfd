import gym
import ppaquette_gym_doom
from doom_utils import ToDiscrete

DOOM_ENV = 'DoomBasic-v0'
env = gym.make('ppaquette/' + DOOM_ENV)
env = ToDiscrete("minimal")(env)

env.reset()

done = False
while not done:
        env.render()
        action = 0
        # action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

# env.close()

[
	"ATTACK",
	"USE",
	"JUMP",
	"CROUCH",
	"TURN180",
	"ALTATTACK",
	"RELOAD",
	"ZOOM",
	"SPEED",
	"STRAFE",
	"MOVE_RIGHT",
	"MOVE_LEFT",
	"MOVE_BACKWARD",
	"MOVE_FORWARD",
	"TURN_RIGHT",
	"TURN_LEFT",
	"LOOK_UP",
	"LOOK_DOWN",
	"MOVE_UP",
	"MOVE_DOWN",
	"LAND",
	"SELECT_WEAPON1",
	"SELECT_WEAPON2",
	"SELECT_WEAPON3",
	"SELECT_WEAPON4",
	"SELECT_WEAPON5",
	"SELECT_WEAPON6",
	"SELECT_WEAPON7",
	"SELECT_WEAPON8",
	"SELECT_WEAPON9",
	"SELECT_WEAPON0",
	"SELECT_NEXT_WEAPON",
	"SELECT_PREV_WEAPON",
	"DROP_SELECTED_WEAPON",
	"ACTIVATE_SELECTED_ITEM",
	"SELECT_NEXT_ITEM",
	"SELECT_PREV_ITEM",
	"DROP_SELECTED_ITEM",
	"LOOK_UP_DOWN_DELTA",
	"TURN_LEFT_RIGHT_DELTA",
	"MOVE_FORWARD_BACKWARD_DELTA",
	"MOVE_LEFT_RIGHT_DELTA",
	"MOVE_UP_DOWN_DELTA"
]