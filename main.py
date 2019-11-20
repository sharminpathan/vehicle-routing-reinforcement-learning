import numpy as np
from environment import *
from agent import *
from keras import backend as K

if __name__ == "__main__":
    # parameters
    epoch = 100
    hidden_size = 100
    env=Env()
    num_actions = env.n_cust #we need idx
    total_reward=0
    
    sess = tf.Session()
    K.set_session(sess)
    agent = Agent(env, sess)


    for e in range(epoch):
        loss = 0.
        env.reset()
        done = False
        
        # get initial input
        input_t, path_t = env.observe()

        while not done:
            env.was_zero=False
            input_tm1 = np.array(input_t)
            path_tm1 = np.array(path_t)
            
            # get next action
            action = agent.act(input_tm1, path_tm1)
            
            # apply action, get rewards and new state
            input_t, path_t, reward, done = env.act(int(np.argmax(action)))
            
            
            if not env.was_zero:
                total_reward=total_reward+reward
                agent.remember([input_tm1, path_tm1, action, reward, input_t, path_t], done)
                
                agent.train()
                
        print("Epoch {:03d}/99 | Loss {:.4f} | Path_length {}".format(e, loss, total_reward))
        
        total_reward=0


