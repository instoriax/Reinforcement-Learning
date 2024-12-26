import argparse
import json
import numpy as np
import requests
import cv2


def connect(agent, url: str = 'http://localhost:5000'):
    frame_stack = None
    while True:
        # Get the observation
        response = requests.get(f'{url}')
        if json.loads(response.text).get('error'):
            print(json.loads(response.text)['error'])
            break
        obs = json.loads(response.text)['observation']
        obs = np.array(obs).astype(np.uint8)

        obs = obs.transpose(1, 2, 0)
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84))

        if frame_stack is None:
            frame_stack = np.stack([obs] * 8, axis=2)
        else:
            frame_stack = np.roll(frame_stack, -1, axis=2)
            frame_stack[:, :, -1] = obs

        # Decide an action based on the observation (Replace this with your RL agent logic)
        # print(obs.shape)
        action_to_take, _ = agent.predict(frame_stack)  # Replace with actual action

        # Send an action and receive new observation, reward, and done status
        response = requests.post(f'{url}', json={'action': action_to_take.tolist()})
        if json.loads(response.text).get('error'):
            print(json.loads(response.text)['error'])
            break

        result = json.loads(response.text)
        terminal = result['terminal']

        if terminal:
            print('Episode finished.')
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, default='http://localhost:5000', help='The url of the server.')
    args = parser.parse_args()


    class RandomAgent:
        def __init__(self, action_space):
            self.action_space = action_space

        def act(self, observation):
            return self.action_space.sample()


    # Initialize the RL Agent
    import gymnasium as gym

    rand_agent = RandomAgent(
        action_space=gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32))

    connect(rand_agent, url=args.url)
