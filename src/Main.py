from src.DataManager import Datamanager
from src.RestrictedBoltzmannMachine import RestrictedBoltzmannMachine

data_manager = Datamanager(training_set_path='../data/ml-100k/u1.base', test_set_path='../data/ml-100k/u1.test')

restrictedBolztmannMachine = RestrictedBoltzmannMachine(hidden_nodes=100,
                                                        number_of_users=data_manager.number_of_users,
                                                        number_of_movies=data_manager.number_of_movies,
                                                        data_manager=data_manager)

restrictedBolztmannMachine.train(epochs=10, batch_size=100)

restrictedBolztmannMachine.test()
