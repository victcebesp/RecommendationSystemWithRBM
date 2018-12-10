import torch


class RestrictedBoltzmannMachine:

    def __init__(self, hidden_nodes, number_of_users, number_of_movies, data_manager):
        self.weights = torch.randn(number_of_movies, hidden_nodes)
        self.hidden_nodes_bias = torch.randn(1, hidden_nodes)
        self.visible_nodes_bias = torch.randn(1, number_of_movies)
        self.number_of_users = number_of_users
        self.number_of_movies = number_of_movies
        self.data_manager = data_manager

    def sample_hidden_nodes(self, x):
        multiplication = torch.mm(x, self.weights)
        activation = multiplication + self.hidden_nodes_bias.expand_as(multiplication)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_visible_nodes(self, y):
        multiplication = torch.mm(y, self.weights.t())
        activation = multiplication + self.visible_nodes_bias.expand_as(multiplication)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def update_parameters(self, initial_visible_nodes, k_visible_nodes, initial_hidden_probability, k_hidden_probability):
        self.weights += torch.mm(initial_visible_nodes.t(), initial_hidden_probability) - \
                        torch.mm(k_visible_nodes.t(), k_hidden_probability)
        self.visible_nodes_bias += torch.sum((initial_visible_nodes - k_visible_nodes), 0)
        self.hidden_nodes_bias += torch.reshape(torch.sum((initial_hidden_probability - k_hidden_probability), 0),
                                                (self.hidden_nodes_bias.shape[0], self.hidden_nodes_bias.shape[1]))

    def train(self, epochs, batch_size):
        for epoch in range(1, epochs + 1):
            loss = 0
            counter = 0.
            for user_id in range(0, self.number_of_users - batch_size, batch_size):
                k_visible_nodes = self.data_manager.training_set[user_id:user_id + batch_size]
                initial_visible_nodes = self.data_manager.training_set[user_id:user_id + batch_size]
                probability_of_hidden_given_visible, _ = self.sample_hidden_nodes(
                    initial_visible_nodes)

                k_visible_nodes = self.gibbs_sampling(initial_visible_nodes, k_visible_nodes)

                k_probability_of_hidden_given_visible, _ = self.sample_hidden_nodes(k_visible_nodes)

                self.update_parameters(initial_visible_nodes,
                                       k_visible_nodes,
                                       probability_of_hidden_given_visible,
                                       k_probability_of_hidden_given_visible)
                loss += self.get_loss_increase(initial_visible_nodes, k_visible_nodes)
                counter += 1.

            print('epoch: ' + str(epoch) + ' -> loss: ' + str(loss / counter))

    def test(self):
        test_loss = 0
        counter = 0.
        for user_id in range(self.number_of_users):
            visible_nodes = self.data_manager.training_set[user_id:user_id + 1]
            target_visible_nodes = self.data_manager.test_set[user_id:user_id + 1]

            if len(target_visible_nodes[target_visible_nodes >= 0]) > 0:
                _, hidden_nodes = self.sample_hidden_nodes(visible_nodes)
                _, visible_nodes = self.sample_visible_nodes(hidden_nodes)
                test_loss += self.get_loss_increase(target_visible_nodes, visible_nodes)
                counter += 1.
        print('test_loss: ' + str(test_loss / counter))

    def gibbs_sampling(self, initial_visible_nodes, k_visible_nodes):
        for k in range(10):
            _, hidden_nodes = self.sample_hidden_nodes(k_visible_nodes)
            _, k_visible_nodes = self.sample_visible_nodes(hidden_nodes)
            k_visible_nodes[initial_visible_nodes < 0] = initial_visible_nodes[initial_visible_nodes < 0]
        return k_visible_nodes

    def get_loss_increase(self, initial_visible_nodes, k_visible_nodes):
        return torch.mean(torch.abs(
            initial_visible_nodes[initial_visible_nodes >= 0] - k_visible_nodes[initial_visible_nodes >= 0]))