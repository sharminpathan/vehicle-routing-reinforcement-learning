import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow import keras
from tensorflow.keras import layers


class Env:
    def __init__(self):
        self.capacity = 15
        self.path = [-1]
        self.depot = [0,0,0]
        self.orders = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5],[6,6,6],[7,7,7],[8,8,8],[9,9,9],[10,10,10]]).astype('float32')
        self.done = False
        self.mask = np.ones(num_actions).astype('float32')
        self.dist = 0
        self.total_dist = 0
        
    def reset(self):
        self.path = [-1]
        self.orders = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5],[6,6,6],[7,7,7],[8,8,8],[9,9,9],[10,10,10]]).astype('float32')
        self.done = False
        self.mask = np.ones(num_actions).astype('float32')
        self.dist = 0
        self.total_dist = 0
        return self.observe()
        
    def observe(self):
        return self.orders.copy(), self.mask.copy()

    def is_over(self):
        return not self.orders[:,0].any()
    
    def get_reward(self, src, dest):
        if src==-1:
            return (abs(self.depot[1] - self.orders[dest,1]) + abs(self.depot[2] - self.orders[dest,2]))
        return (abs(self.orders[src,1] - self.orders[dest,1]) + abs(self.orders[src,2] - self.orders[dest,2]))
        
    def step(self, order_id):
        self.orders[order_id,0] = 0
        self.dist = 50-self.get_reward(self.path[-1],order_id)
        self.total_dist = self.total_dist - self.get_reward(self.path[-1],order_id)
        self.path.append(order_id)
        self.mask[order_id] = 0
        return self.orders.copy(), self.mask.copy(), self.dist, self.is_over()


# Configuration paramaters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 10000
num_actions = 10

env = Env()


def create_q_model():
    inputs = layers.Input(shape=(env.orders.shape))
    mask = layers.Input(shape=(num_actions,))
    
    layer1 = layers.Conv1D(32, 3, activation="relu")(inputs)
    layer2 = layers.Flatten()(layer1)
    layer3 = layers.Dense(512, activation="relu")(layer2)
    layer4 = layers.Dense(num_actions, activation="linear")(layer3)
    action = layers.Multiply()([layer4, mask])
    return keras.Model(inputs=[inputs, mask], outputs=action)

    
# The first model makes the predictions for Q-values which are used to
# make a action.
model = create_q_model()
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_target = create_q_model()


decay_learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=.001, decay_steps=1500, decay_rate=.1)
optimizer = keras.optimizers.Adam(learning_rate=decay_learning_rate, clipnorm=1.0)

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
mask_history = []
mask_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0
# Number of frames to take random action and observe output
epsilon_random_frames = 50000
# Number of frames for exploration
epsilon_greedy_frames = 1000000.0
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 100000
# Train the model after 4 actions
update_after_actions = 4
# How often to update the target network
update_target_network = 10000
# Using huber loss for stability
loss_function = keras.losses.Huber()

while True:  # Run until solved
    state, mask = env.reset()
    episode_reward = 0

    for timestep in range(1, max_steps_per_episode):
        frame_count += 1

        # Use epsilon-greedy for exploration
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(np.where(env.mask==1)[0])
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            mask_tensor = tf.convert_to_tensor(mask)
            mask_tensor = tf.expand_dims(mask_tensor, 0)
            action_probs = model([state_tensor, mask_tensor], training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()
            if env.mask[action]==0: 
                action = np.random.choice(np.where(env.mask==1)[0])

        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # Apply the sampled action in our environment
        state_next, mask_next, reward, done = env.step(action)
        state_next = np.array(state_next)
        mask_next = np.array(mask_next)

        episode_reward += reward

        # Save actions and states in replay buffer
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        mask_history.append(mask)
        mask_next_history.append(mask_next)
        done_history.append(done)
        rewards_history.append(reward)
        state = state_next
        mask = mask_next

        # Update every fourth frame and once batch size is over 32
        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:

            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(done_history)), size=batch_size)

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            mask_sample = np.array([mask_history[i] for i in indices])
            mask_next_sample = np.array([mask_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(done_history[i]) for i in indices]
            )

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = model_target.predict([state_next_sample, mask_next_sample])
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * tf.reduce_max(
                future_rewards, axis=1
            )

            # If final frame set the last value to -1
            # updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model([state_sample, mask_sample])

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            ### pred
            state, mask = env.reset()
            done=False
            count_random=0
            while not done:
                action_probs = model.predict([np.expand_dims(state, axis=0), np.expand_dims(mask, axis=0)])
                action = np.argmax(action_probs)
                if env.mask[action]==0: 
                    count_random += 1
                    action = np.random.choice(np.where(env.mask==1)[0])
                # Apply the sampled action in our environment
                state, mask, reward, done = env.step(action)
            print('count_random:', count_here, 'total distance: ',np.abs(env.total_dist))

        if frame_count % update_target_network == 0:
            # update the the target network with new weights
            model_target.set_weights(model.get_weights())
            
        # Limit the state and reward history
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del mask_history[:1]
            del mask_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if done:
            break

