import gym
from rl_brain import DeepQNetwork

def run_env():
    step = 0
    for episode in range(10):
        observation = env.reset()

        while True:
            env.render()
            action = RL.choose_action(observation)
            observation_,reward,done,info = env.step(action)
            position,velocity = observation_
            reward = abs(position-(-0.5))
            RL.store_transition(observation,action,reward,observation_)
            if (step > 1000) and (step % 5 == 0):
                RL.learn()
            observation = observation_
            if done:
                break
            step+=1
        RL.save()
        print("game over")

if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    env = env.unwrapped
    RL = DeepQNetwork(n_actions=3,n_features=2,
                      learning_rate = 0.001,
                      reward_decay = 0.9,
                      e_greedy=0.9,
                      replace_target_iter=300,
                      memory_size=3000,
                      )
    run_env()
    RL.plot_cost()