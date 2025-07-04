import gym
import numpy as np
from MLSH_agent import MLSH_agent
from Discrete_SAC_Agent import SACAgent
import torch
import random
import os
WARMUP_PERIOD = 60
TRAIN_PERIOD =  30
TRAINING_EVALUATION_RATIO = WARMUP_PERIOD + TRAIN_PERIOD + 1
RUNS = 5
EPISODES_PER_RUN = TRAINING_EVALUATION_RATIO * 300
STEPS_PER_EPISODE = 40
NUM_SUBPOLICIES = 6
SEED=42
MASTER_LEN=3
N_PARRA=5
gym.register(
    id='DialogueEnv-v0',
    entry_point='DialogueEnvs.DialogueEnvMIparra:DialogueEnvMI'
)

if __name__ == "__main__":
    env = gym.make('DialogueEnv-v0',n_parra=N_PARRA)

    agent_results = []
    agent_tasks = []
    name = 'DREAM_no_maml'
    use_maml = False

    for run in range(RUNS):
        agent = MLSH_agent(env,NUM_SUBPOLICIES,master_len_replay_buffer=50000,sub_len_replay_buffer=50000,master_replay_mini_batch_size=1000,sub_replay_mini_batch_size=1000,master_learning_rate=10**-3,sub_learning_rate=10**-4,master_mbpo=False,use_maml=use_maml)
        run_results = []
        eval_reward=0
        tasks = []
        master_actions = []
        master_states = []
        task_order = np.random.permutation(3)
        env.set_task(task_order[0])
        id_task=0
        task = task_order[0]
        for episode_number in range(EPISODES_PER_RUN):
            try:
                os.mkdir('agent_mbpo'+name+'_'+str(episode_number))
                os.mkdir('agent_mbpo'+name+'_'+str(episode_number)+'/master')
                os.mkdir('agent_mbpo'+name+'_'+str(episode_number)+'/subpolicies')
            except:
                pass
            print('\r', f'********************************Run: {run + 1}/{RUNS} | Episode: {episode_number + 1}/{EPISODES_PER_RUN} | Evaluation reward: {np.mean(eval_reward)}', end=' ')
            evaluation_episode = episode_number % TRAINING_EVALUATION_RATIO == WARMUP_PERIOD
            warmup_episode = episode_number % TRAINING_EVALUATION_RATIO < WARMUP_PERIOD
            episode_reward = 0
            state = env.reset()

            master_state = env.ob_master()
            last_state_master=master_state
            done = False
            dones = [False for i in range(N_PARRA)]
            i = 0
            subpolicy=[0 for i in range(N_PARRA)]
            master_reward=0
            if evaluation_episode:
                agent.master_policy.replay_buffer.empty()
            subpolicy = agent.get_next_master_action(master_state, evaluation_episode=evaluation_episode)

            while not done and i < STEPS_PER_EPISODE:
                i += 1
                if i % MASTER_LEN == 0:
                    if not evaluation_episode:
                        if warmup_episode:
                            agent.train_on_master_transition(last_state_master, subpolicy, master_state, master_reward, dones)
                        else:
                            agent.master_policy.replay_buffer.empty()
                    else:
                        agent.add_master_transition(last_state_master, subpolicy, master_reward, master_state, dones)
                    last_state_master = master_state
                    master_reward=0
                    subpolicy = agent.get_next_master_action(master_state, evaluation_episode=(evaluation_episode or  not warmup_episode))
                master_actions.append(subpolicy)
                master_states.append(master_state)
                action = agent.get_next_action(state,subpolicy, evaluation_episode=(evaluation_episode or warmup_episode))
                next_master_state,next_state, reward, dones, info= env.step(action)
                master_reward += reward
                if not evaluation_episode and not warmup_episode:
                    agent.train_on_transition(state, action, next_state, reward, dones,subpolicy)

                elif not evaluation_episode and warmup_episode:
                    for i in range(len(agent.subpolicies.replay_buffer)):

                        agent.subpolicies.replay_buffer[i].empty()

                else:
                    episode_reward += reward
                state = next_state
                master_state = next_master_state


                done = True
                for d in dones:
                    if not d:
                        done = False

            if evaluation_episode:
                run_results.append(np.mean(episode_reward))
                eval_reward = episode_reward
                np.save('agent_results_mbpo'+name+'.npy', run_results)
                tasks.append(task)
                np.save('tasks_mbpo'+name+'.npy', tasks)
                agent.save_agent('agent_mbpo'+name+'_'+str(episode_number))
                np.save("conv_"+name+'_'+str(episode_number),env.conv)
                np.save("master_actions_"+name+'_'+str(episode_number),master_actions)
                np.save("master_states_"+name+'_'+str(episode_number),master_states)
                id_task +=1
                if use_maml:

                    agent.master_policy.update_meta()
                    agent.master_policy.clone()
                    agent.master_policy.replay_buffer.empty()
                if id_task%3==0:
                    task_order = np.random.permutation(3)
                task =task_order[id_task%3]
                env.set_task(task)


        agent_results.append(run_results)
        agent_tasks.append(tasks)
        n_results = EPISODES_PER_RUN // TRAINING_EVALUATION_RATIO
        results_mean = [np.mean([agent_result[n] for agent_result in agent_results]) for n in range(n_results)]
        results_std = [np.std([agent_result[n] for agent_result in agent_results]) for n in range(n_results)]
        mean_plus_std = [m + s for m, s in zip(results_mean, results_std)]
        mean_minus_std = [m - s for m, s in zip(results_mean, results_std)]

        x_vals = list(range(len(results_mean)))
        x_vals = [x_val * (TRAINING_EVALUATION_RATIO - 1) for x_val in x_vals]

        np.save('results_mean_local_mbpo'+name+'.npy', results_mean)
        np.save('results_std_local_mbpo'+name+'.npy', results_std)
        np.save('x_vals_mbpo'+name+'.npy', x_vals)
        np.save('tasks_mbpo'+name+'.npy', agent_tasks)
        np.save('agent_results_mbpo'+name+'.npy', agent_results)
        agent.save_agent('agent_mbpo'+name+'_'+str(episode_number))

    env.close()

    n_results = EPISODES_PER_RUN // TRAINING_EVALUATION_RATIO
    results_mean = [np.mean([agent_result[n] for agent_result in agent_results]) for n in range(n_results)]
    results_std = [np.std([agent_result[n] for agent_result in agent_results]) for n in range(n_results)]
    mean_plus_std = [m + s for m, s in zip(results_mean, results_std)]
    mean_minus_std = [m - s for m, s in zip(results_mean, results_std)]

    x_vals = list(range(len(results_mean)))
    x_vals = [x_val * (TRAINING_EVALUATION_RATIO - 1) for x_val in x_vals]
    np.save('results_mean_local_mbpo'+name+'.npy', results_mean)
    np.save('results_std_local_mbpo'+name+'.npy', results_std)
    np.save('x_vals_mbpo'+name+'.npy',x_vals)
    np.save('agent_results_mbpo'+name+'.npy', agent_results)
    agent.save_agent('agent_mbpo'+name+'_'+str(episode_number))