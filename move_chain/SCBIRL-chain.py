import jax.numpy as np
from jax import grad, jit, value_and_grad
from jax import random
from jax.example_libraries import optimizers
import jax
import haiku as hk
import os

from tqdm import tqdm
import numpy as onp
import pickle

from utils import *


def hidden_layers(layers=1, units=64):
    hidden = []
    for i in range(layers):
        hidden += [hk.Linear(units), jax.nn.elu]
    return hidden


def encoder_model(inputs, layers=2, units=64, state_only=True, a_dim=None):
    """
    Create an encoder model for fitting posterior probabilities of a reward function.

    Args:
        inputs (jax.numpy):The input tensor representing the state.
        layers (int, optional): The number of hidden layers in the encoder model. Defaults to 2.
        units (int, optional): The number of units (neurons) in each hidden layer. Defaults to 64.
        state_only (bool, optional): If True, encode only the state information; if False, encode state-action pairs. Defaults to True.
        a_dim (int, optional): The dimension of the action space (only needed if state_only is False). Defaults to None.

    Returns:
        outputs posterior probabilities for a reward function. mean and variance
    """    
    out_dim = 2
    if not state_only:
        out_dim = a_dim * 2
    mlp = hk.Sequential(hidden_layers(layers) + [hk.Linear(out_dim)])
    return mlp(inputs)


def q_network_model(inputs, a_dim, layers=2, units=64):
    """
    Create a Q-network model for reinforcement learning.

    Args:
        inputs (tf.Tensor): The input tensor representing the state.
        a_dim (int): The number of actions in the action space.
        layers (int, optional): The number of hidden layers in the Q-network. Defaults to 2.
        units (int, optional): The number of units (neurons) in each hidden layer. Defaults to 64.

    Returns:
        A Q-network model that takes the state as input and outputs Q-values for each action.
    """
    mlp = hk.Sequential(hidden_layers(layers) + [hk.Linear(a_dim)])
    return mlp(inputs)


def klGaussianStandard(mean, var):
    return 0.5 * (-np.log(var) - 1.0 + var + mean ** 2)

def kl_divergence(mean1, stddev1, mean2, stddev2):
    """
    Calculate the Kullback-Leibler (KL) divergence between two one-dimensional Gaussian distributions.

    Args:
        mean1 (float): Mean of the first distribution.
        stddev1 (float): Standard deviation of the first distribution.
        mean2 (float): Mean of the second distribution.
        stddev2 (float): Standard deviation of the second distribution.

    Returns:
        float: The KL divergence value.
    """
    # Calculate the KL divergence
    kl = np.log(stddev2 / stddev1) + ((stddev1 ** 2 + (mean1 - mean2) ** 2) / (2 * stddev2 ** 2)) - 0.5
    return kl


class avril:
    """
    Class for implementing the AVRIL algorithm of Chan and van der Schaar (2021).
    This model is designed to be instantiated before calling the .train() method
    to fit to data.
    """

    def __init__(
        self,
        inputs: np.array,
        targets: np.array,
        state_dim: int,
        action_dim: int,
        state_only: bool = True,
        encoder_layers: int = 2,
        encoder_units: int = 64,
        decoder_layers: int = 2,
        decoder_units: int = 64,
        seed: int = 41310,
    ):
        """
        Parameters
        ----------

        inputs: np.array
            State training data of size [num_pairs x 2 x state_dimension]
        targets: np.array
            Action training data of size [num_pairs x 2 x 1]
        state_dim: int
            Dimension of state space
        action_dim: int
            Size of action space
        state_only: bool, True
            Whether learnt reward is state-only (as opposed to state-action)
        encoder_layers: int, 2
            Number of hidden layers in encoder network
        encoder_units: int, 64
            Number of hidden units per layer of encoder network
        decoder_layers: int, 2
            Number of hidden layers in decoder network
        decoder_units: int, 64
            Number of hidden units per layer of decoder network
        seed: int, 41310
            Random seed - required for JAX PRNG to work
        """

        self.key = random.PRNGKey(seed)

        self.encoder = hk.transform(encoder_model)
        self.q_network = hk.transform(q_network_model)

        self.inputs = inputs
        self.targets = targets
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.state_only = state_only
        self.encoder_layers = encoder_layers
        self.encoder_units = encoder_units
        self.decoder_layers = decoder_layers
        self.decoder_units = decoder_units

        self.e_params = self.encoder.init(
            self.key, inputs, encoder_layers, encoder_units, self.state_only, action_dim
        )
        self.q_params = self.q_network.init(
            self.key, inputs, action_dim, decoder_layers, decoder_units
        )

        self.params = (self.e_params, self.q_params)

        self.load_params = False
        self.pre_params = None
        return

    def predict(self, state):
        #  Returns predicted action logits for a given state
        logit = self.q_network.apply(
            self.q_params,
            self.key,
            state,
            self.a_dim,
            self.decoder_layers,
            self.decoder_units,
        )
        return logit

    def reward(self, state):
        #  Returns reward function parameters for a given state
        r_par = self.encoder.apply(
            self.e_params,
            self.key,
            state,
            self.encoder_layers,
            self.encoder_units,
            self.state_only,
            self.a_dim,
        )
        return r_par

    def rewardValue(self,state):
        """
            return the given reward of a given state
        Args:
            state (np.array): attribute of the given state

        Returns:
            (int): reward value
        """        
        mean, log_variance = self.reward(state) # mean and log variance
        sample_size=1
        sample_reward = onp.random.normal(mean, onp.abs(np.exp(log_variance)), sample_size)
        return sample_reward

    def QValue(self, state):
        q_values = self.q_network.apply(
            self.q_params,
            self.key,
            state,
            self.a_dim,
            self.decoder_layers,
            self.decoder_units,
        )
        return state
    
    def modelSave(self,model_save_path):
        with open(model_save_path,'wb') as f:
            print("save params to {}!".format(model_save_path))
            pickle.dump(self.params, f, protocol=pickle.HIGHEST_PROTOCOL)

    def elbo(self, params, key, inputs, targets):
        """
        Method for calculating ELBO

        Parameters
        ----------

        params: tuple
            JAX object containing parameters of the model
        key:
            JAX PRNG key
        inputs: np.array
            State training data of size [num_pairs x 2 x state_dimension]
        targets: np.array
            Action training data of size [num_pairs x 2 x 1]

        Returns
        -------

        elbo: float
            Value of the ELBO
        """

        # Calculate Q-values for current state
        e_params, q_params = params
        q_values = self.q_network.apply(
            q_params,
            key,
            inputs[:, 0, :],
            self.a_dim,
            self.decoder_layers,
            self.decoder_units,
        )
        q_values_a = np.take_along_axis(
            q_values, targets[:, 0, :].astype(np.int32), axis=1
        ).reshape(len(inputs))

        # Calculate Q-values for next state
        q_values_next = self.q_network.apply(
            q_params,
            key,
            inputs[:, 1, :],
            self.a_dim,
            self.decoder_layers,
            self.decoder_units,
        )
        q_values_next_a = np.take_along_axis(
            q_values_next, targets[:, 1, :].astype(np.int32), axis=1
        ).reshape(len(inputs))

        # Calaculate TD error
        td = q_values_a - q_values_next_a

        def getRewardParameters(encoder_params):
            r_par = self.encoder.apply(
                encoder_params,
                key,
                inputs[:, 0, :],
                self.encoder_layers,
                self.encoder_units,
                self.state_only,
                self.a_dim,
            )
            if self.state_only:
                means = r_par[:, 0].reshape(len(inputs))  # mean
                log_sds = r_par[:, 1].reshape(len(inputs))  # log std var
            else:
                means = np.take_along_axis(
                    r_par, (targets[:, 0, :]).astype(int), axis=1
                ).reshape((len(inputs),))
                log_sds = np.take_along_axis(
                    r_par, (a_dim + targets[:, 0, :]).astype(int), axis=1
                ).reshape((len(inputs),))
            return means, log_sds
        
        means, log_sds = getRewardParameters(e_params)
        if self.load_params:
            e_params_pre, _ = self.pre_params
            means_pre, log_sds_pre = getRewardParameters(e_params_pre)
            
            kl = kl_divergence(means, np.exp(log_sds), means_pre, np.exp(log_sds_pre)).mean()
        else:
            kl = klGaussianStandard(means, np.exp(log_sds) ** 2).mean()

        # NOTE: Add a negative sign in front of each formula to solve for the minimum value
        # Calculate log-likelihood of TD error given reward parameterisation
        lambda_value = 1
        irl_loss = -jax.scipy.stats.norm.logpdf(td, means, np.exp(log_sds)).mean()

        # Calculate log-likelihood of actions
        pred = jax.nn.log_softmax(q_values)
        neg_log_lik = -np.take_along_axis(
            pred, targets[:, 0, :].astype(np.int32), axis=1
        ).mean()

        return neg_log_lik + kl + lambda_value*irl_loss

    def loadParams(self,model_path):
        print("load params from {}!".format(model_path))
        with open(model_path, 'rb') as f:
            self.params = pickle.load(f)     
            self.load_params = True
            self.pre_params = self.params

    def train(self, iters: int = 1000, batch_size: int = 64, l_rate: float = 1e-4):
        """
        Training function for the model.

        Parameters
        ----------
        iters: int, 1000
            Number of training update steps (NOTE: Not epochs)
        batch_size: int, 64
            Batch size for stochastic optimisation
        l_rate: float, 1e-4
            Main learning rate for Adam
        """

        inputs = self.inputs
        targets = self.targets

        init_fun, update_fun, get_params = optimizers.adam(l_rate)
        update_fun = jit(update_fun)
        get_params = jit(get_params)

        params = self.params

        param_state = init_fun(params)

        loss_grad = jit(value_and_grad(self.elbo))

        len_x = len(self.inputs[:, 0, :])
        num_batches = np.ceil(len_x / batch_size)

        indx_list = np.array(range(len_x))

        key = self.key

        for itr in tqdm(range(iters)):

            if itr % num_batches == 0:
                indx_list_shuffle = jax.random.permutation(key, indx_list)

            indx = int((itr % num_batches) * batch_size)
            indxes = indx_list_shuffle[indx : (batch_size + indx)]

            key, subkey = random.split(key)

            lik, g_params = loss_grad(params, key, inputs[indxes], targets[indxes])

            param_state = update_fun(itr, g_params, param_state)

            params = get_params(param_state)

        self.e_params = params[0]
        self.q_params = params[1]

        self.params = params

def computeRewardOrValue(model, input_path, output_path, attribute_type='value'):
    """
    Compute rewards or state values for each state using a given model and save to a CSV file.

    Parameters:
    - model: The trained model. Should have a method `rewardValue` for computing rewards and `QValue` for computing Q values.
    - input_path (str): Path to the input CSV file containing states.
    - output_path (str): Path to save the output CSV file with computed attributes (rewards or state values).
    - attribute_type (str): Either 'value' to compute state values or 'reward' to compute rewards.

    Returns:
    None. Writes results to the specified output CSV file.
    """
    
    state_attribute = pd.read_csv(input_path)
    state_attribute[attribute_type] = 0

    if attribute_type == 'value':
        for index, row in tqdm(state_attribute.iterrows(), total=len(state_attribute)):
            state = np.array(row.values[1:31])
            q_values = model.QValue(state)
            state_attribute.iloc[index, -1] = np.max(q_values)
    elif attribute_type == 'reward':
        for index, row in tqdm(state_attribute.iterrows(), total=len(state_attribute)):
            state = np.array(row.values[1:31])
            r = sum(np.abs(model.rewardValue(state)) for _ in range(10)) / 10
            state_attribute.iloc[index, -1] = r[0]
    else:
        raise ValueError("attribute_type should be either 'value' or 'reward'.")
    
    if output_path is not None:
        state_attribute.to_csv(output_path, index=False)
    
    return

def afterMigrt(after_migrt_file,input_path,output_path,model):
    # load params from pre cognitive map
    model.loadParams('./model/params.pickle')
    # read traj chain file
    with open(after_migrt_file, 'r') as file:
        loaded_dicts_list1 = json.load(file)
    traj_chains = [TravelData(**d) for d in loaded_dicts_list1]

    state_attribute = pd.read_csv(input_path)
    results_df = pd.DataFrame()
    results_df['fnid'] = state_attribute['fnid']
    pre_date = 0

    for tc in traj_chains:
        # Based on the travel chain of each day, update the model parameters.
        date = tc.date
        state_next_state = []
        action_next_action = []
        
        for t in range(len(tc.travel_chain)-1):
            s_n_s = onp.zeros((2, s_dim))
            this_state,next_state=tc.travel_chain[t],tc.travel_chain[t+1]
            row = state_attribute[state_attribute['fnid'] == this_state]
            s_n_s[0, :] = np.array(row.values[0][1:31])
            row = state_attribute[state_attribute['fnid'] == next_state]
            s_n_s[1, :] = np.array(row.values[0][1:31])
            state_next_state.append(s_n_s)

            a_n_a = onp.zeros((2,1))
            a_n_a[0] = tc.id_chain[t+1]
            try:
                a_n_a[1] = tc.id_chain[t+2]
            except IndexError:
                a_n_a[1] = -1 
            action_next_action.append(a_n_a)

        state_next_state = np.array(onp.array(state_next_state))
        action_next_action = np.array(onp.array(action_next_action))
        model.inputs = state_next_state        
        model.targets = action_next_action

        if pre_date != 0:
            model_path = os.path.join("./data/after_migrt/model", f"{pre_date}.pickle")
            model.loadParams(model_path)
        model.train(iters=5000)
        model_save_path = "./data/after_migrt/model/" +str(date)+".pickle"
        model.modelSave(model_save_path)

        reward_values = []
        for index, row in tqdm(state_attribute.iterrows(), total=len(state_attribute)):
            state = np.array(row.values[1:31])
            r = sum(np.abs(model.rewardValue(state)) for _ in range(100)) / 100
            reward_values.append(r[0])
        results_df[str(date)] = reward_values

        pre_date = date
    results_df.to_csv(output_path)

if __name__ == "__main__":
    path = f'./data/before_migrt.json'
    full_traj_path = f'./data/all_traj.json'
    inputs, targets, a_dim, s_dim =loadTrajChain(path,full_traj_path)
    model = avril(inputs, targets, s_dim, a_dim, state_only=True) # initialization

    # NOTE: model train 
    model.train(iters=50000)
    model_save_path = f'./model/params.pickle'
    model.modelSave(model_save_path)

    # NOTE: model reward before migration
    computeRewardOrValue(model, './data/before_migrt_fnid.csv', './data/before_migrt_reward.csv', attribute_type='reward')
    # computeRewardOrValue(model, './data/all_traj_fnid.csv', './data/all_traj_value.csv', attribute_type='value')

    # NOTE: mode parameters after migration per day
    after_migrt_file = f'./data/after_migrt.json'
    input_path = './data/all_traj_fnid.csv'
    output_path = './data/after_migrt/after_migrt_reward.csv'
    afterMigrt(after_migrt_file,input_path,output_path,model)