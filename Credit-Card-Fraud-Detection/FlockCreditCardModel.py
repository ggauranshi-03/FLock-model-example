#########################ORIGINAL##################################################################
# import json
# from torch import nn
# from torch.utils.data import DataLoader, TensorDataset
# import io
# import torch
# from pandas import DataFrame
# from flock_sdk import FlockSDK, FlockModel
# from models.CreditFraudNetMLP import CreditFraudNetMLP

# class ExampleTorchModal(FlockModel):
#     def __init__(self, features, epochs=1, lr=0.3) -> None:
#         self.epoch = epochs
#         self.features = features
#         self.lr = lr

#         if torch.cuda.is_available():
#             device = torch.device("cuda")
#         else:
#             device = torch.device("cpu")
#         self.device = torch.device(device)

#     def init_dataset(self, dataset_path: str) -> None:
#         self.dataset_path = dataset_path
#         with open(dataset_path, "r") as f:
#             dataset = json.load(f)
#         dataset_df = DataFrame.from_records(dataset)

#         batch_size = 128

#         X_df = dataset_df.iloc[:, :-1]
#         y_df = dataset_df.iloc[:, -1]

#         X_tensor = torch.tensor(X_df.values, dtype=torch.float32)
#         y_tensor = torch.tensor(y_df.values, dtype=torch.float32)

#         y_tensor = y_tensor.unsqueeze(1)
#         dataframe_in_dataset = TensorDataset(X_tensor, y_tensor)

#         self.train_data_loader = DataLoader(
#             dataframe_in_dataset,
#             batch_size=batch_size,
#             shuffle=True,
#             drop_last=True,
#         )
#         self.test_data_loader = DataLoader(
# 	        dataframe_in_dataset,
# 	        batch_size=batch_size,
# 	        shuffle=True,
# 	        drop_last=False,
#         )

#     def train(self, parameters) -> bytes:
#         model = CreditFraudNetMLP(self.features, 1)
#         if parameters != None:
#             model.load_state_dict(torch.load(io.BytesIO(parameters)))
#             model.train()
#             optimizer = torch.optim.SGD(
#                 model.parameters(),
#                 lr=self.lr,
#             )
#             criterion = nn.BCELoss()
#             model.to(self.device)

#             for epoch in range(self.epoch):
#                 train_loss = 0.0
#                 train_correct = 0
#                 train_total = 0
#                 for inputs, targets in self.train_data_loader:
#                     optimizer.zero_grad()

#                     inputs, targets = input.to(self.device), targets.to(self.device)
#                     outputs = model(inputs)

#                     loss = criterion(outputs, targets)
#                     loss.backward()

#                     optimizer.step()

#                     train_loss += loss.item() * inputs.size(0)
#                     predicted = torch.round(outputs).squeeze()
#                     train_total += targets.size(0)
#                     train_correct += (predicted == targets.squeeze()).sum().item()

#             buffer = io.BytesIO()
#             torch.save(model.state_dict(), buffer)
#             return buffer.getvalue()

#     def evaluate(self, parameters) -> float:
#         criterion = nn.BCELoss()

#         model = CreditFraudNetMLP(self.features, 1)
#         if parameters != None:
#             model.load_state_dict(torch.load(io.BytesIO(parameters)))
#         model.to(self.device)
#         model.eval()

#         test_loss = 0.0
#         test_correct = 0
#         test_total = 0

#         with torch.no_grad():
#             for inputs, targets in self.test_data_loader:
#                 inputs, targets = inputs.to(self.device), targets.to(self.device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, targets)

#                 test_loss += loss.item() * inputs.size(0)
#                 predicted = torch.round(outputs).squeeze()
#                 test_total += targets.size(0)
#                 test_correct += (predicted == targets.squeeze()).sum().item()

#         accuracy = test_correct / test_total
#         return accuracy

#     def aggregate(self, parameters_list: list[bytes]) -> bytes:
#         parameters_list = [torch.load(io.BytesIO(parameters)) for parameters in parameters_list]
#         averaged_parames_template = parameters_list[0]

#         for k in averaged_parames_template.keys():
#             temp_w = []
#             for local_w in parameters_list:
#                 temp_w.append(local_w[k])
#             averaged_parames_template[k] = sum(temp_w) / len(temp_w)

#         buffer = io.BytesIO()
#         torch.save(averaged_parames_template, buffer)
#         aggregated_parameters = buffer.getvalue()
#         return aggregated_parameters


# import io

# if __name__ == "__main__":
#     epochs = 1
#     lr = 0.000001
#     features = 30
#     model = ExampleTorchModal(features, epochs=epochs, lr=lr)
#     sdk = FlockSDK(model)
#     sdk.run()
#########################AFTER REMOVING SDK#############################################################
# import io
# import json

# import torch
# from models.CreditFraudNetMLP import CreditFraudNetMLP
# from pandas import DataFrame
# from torch import nn
# from torch.utils.data import DataLoader, TensorDataset


# class ExampleTorchModal:
#     def __init__(self, features, dataset_path, epochs=1, lr=0.3) -> None:
#         self.epoch = epochs
#         self.features = features
#         self.lr = lr
#         self.dataset_path = dataset_path

#         if torch.cuda.is_available():
#             device = torch.device("cuda")
#         else:
#             device = torch.device("cpu")
#         self.device = torch.device(device)

#         # Initialize the dataset
#         self.init_dataset(self.dataset_path)

#     def init_dataset(self, dataset_path: str) -> None:
#         self.dataset_path = dataset_path
#         with open(dataset_path, "r") as f:
#             dataset = json.load(f)
#         dataset_df = DataFrame.from_records(dataset)

#         batch_size = 128

#         X_df = dataset_df.iloc[:, :-1]
#         y_df = dataset_df.iloc[:, -1]

#         X_tensor = torch.tensor(X_df.values, dtype=torch.float32)
#         y_tensor = torch.tensor(y_df.values, dtype=torch.float32)

#         y_tensor = y_tensor.unsqueeze(1)
#         dataframe_in_dataset = TensorDataset(X_tensor, y_tensor)

#         self.train_data_loader = DataLoader(
#             dataframe_in_dataset,
#             batch_size=batch_size,
#             shuffle=True,
#             drop_last=True,
#         )
#         self.test_data_loader = DataLoader(
#             dataframe_in_dataset,
#             batch_size=batch_size,
#             shuffle=True,
#             drop_last=False,
#         )

#     def train(self, parameters=None) -> bytes:
#         model = CreditFraudNetMLP(self.features, 1)
#         if parameters is not None:
#             model.load_state_dict(torch.load(io.BytesIO(parameters)))
#         model.train()
#         optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
#         criterion = nn.BCELoss()
#         model.to(self.device)

#         for epoch in range(self.epoch):
#             train_loss = 0.0
#             train_correct = 0
#             train_total = 0
#             for inputs, targets in self.train_data_loader:
#                 optimizer.zero_grad()
#                 inputs, targets = inputs.to(self.device), targets.to(self.device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, targets)
#                 loss.backward()
#                 optimizer.step()
#                 train_loss += loss.item() * inputs.size(0)
#                 predicted = torch.round(outputs).squeeze()
#                 train_total += targets.size(0)
#                 train_correct += (predicted == targets.squeeze()).sum().item()

#         buffer = io.BytesIO()
#         torch.save(model.state_dict(), buffer)
#         return buffer.getvalue()

#     def evaluate(self, parameters) -> float:
#         criterion = nn.BCELoss()
#         model = CreditFraudNetMLP(self.features, 1)
#         if parameters is not None:
#             model.load_state_dict(torch.load(io.BytesIO(parameters)))
#         model.to(self.device)
#         model.eval()

#         test_loss = 0.0
#         test_correct = 0
#         test_total = 0

#         with torch.no_grad():
#             for inputs, targets in self.test_data_loader:
#                 inputs, targets = inputs.to(self.device), targets.to(self.device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, targets)
#                 test_loss += loss.item() * inputs.size(0)
#                 predicted = torch.round(outputs).squeeze()
#                 test_total += targets.size(0)
#                 test_correct += (predicted == targets.squeeze()).sum().item()

#         accuracy = test_correct / test_total
#         return accuracy

#     def aggregate(self, parameters_list) -> bytes:
#         parameters_list = [
#             torch.load(io.BytesIO(parameters)) for parameters in parameters_list
#         ]
#         averaged_parames_template = parameters_list[0]

#         for k in averaged_parames_template.keys():
#             temp_w = []
#             for local_w in parameters_list:
#                 temp_w.append(local_w[k])
#             averaged_parames_template[k] = sum(temp_w) / len(temp_w)

#         buffer = io.BytesIO()
#         torch.save(averaged_parames_template, buffer)
#         aggregated_parameters = buffer.getvalue()
#         return aggregated_parameters


# if __name__ == "__main__":
#     epochs = 1
#     lr = 0.000001
#     features = 30
#     dataset_path = "data/creditCard.json"

#     model = ExampleTorchModal(features, dataset_path, epochs=epochs, lr=lr)

#     # Training the model
#     trained_parameters = model.train()

#     # Evaluating the model
#     accuracy = model.evaluate(trained_parameters)
#     print(f"Model accuracy: {accuracy * 100:.2f}%")

#     # For aggregation example, you need a list of parameters from different training sessions
#     # Since we don't have multiple sessions, we'll simulate it with the same parameters
#     parameters_list = [trained_parameters, trained_parameters]  # Example list
#     aggregated_parameters = model.aggregate(parameters_list)

#     # Re-evaluate the aggregated parameters
#     aggregated_accuracy = model.evaluate(aggregated_parameters)
#     print(f"Aggregated model accuracy: {aggregated_accuracy * 100:.2f}%")

#########################AFTER REMOVING SDK & ADDING WAND################################################
import io
import json

import torch
import wandb
from models.CreditFraudNetMLP import CreditFraudNetMLP
from pandas import DataFrame
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ExampleTorchModal:
    def __init__(self, features, dataset_path, epochs=1, lr=0.3) -> None:
        self.epoch = epochs
        self.features = features
        self.lr = lr
        self.dataset_path = dataset_path

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.device = torch.device(device)

        # Initialize the dataset
        self.init_dataset(self.dataset_path)

        # Initialize Weights & Biases
        wandb.init(project="Credit_Card_Flock")

    def init_dataset(self, dataset_path: str) -> None:
        self.dataset_path = dataset_path
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
        dataset_df = DataFrame.from_records(dataset)

        batch_size = 128

        X_df = dataset_df.iloc[:, :-1]
        y_df = dataset_df.iloc[:, -1]

        X_tensor = torch.tensor(X_df.values, dtype=torch.float32)
        y_tensor = torch.tensor(y_df.values, dtype=torch.float32)

        y_tensor = y_tensor.unsqueeze(1)
        dataframe_in_dataset = TensorDataset(X_tensor, y_tensor)

        self.train_data_loader = DataLoader(
            dataframe_in_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        self.test_data_loader = DataLoader(
            dataframe_in_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

    def train(self, parameters=None) -> bytes:
        model = CreditFraudNetMLP(self.features, 1)
        if parameters is not None:
            model.load_state_dict(torch.load(io.BytesIO(parameters)))
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()
        model.to(self.device)

        for epoch in range(1, self.epoch + 1):
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for inputs, targets in self.train_data_loader:
                optimizer.zero_grad()
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                predicted = torch.round(outputs).squeeze()
                train_total += targets.size(0)
                train_correct += (predicted == targets.squeeze()).sum().item()

            # Log loss and accuracy using wandb
            wandb.log(
                {
                    "epoch": epoch,
                    "loss": train_loss / train_total,
                    "accuracy": train_correct / train_total,
                }
            )

        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        return buffer.getvalue()

    def evaluate(self, parameters) -> float:
        criterion = nn.BCELoss()
        model = CreditFraudNetMLP(self.features, 1)
        if parameters is not None:
            model.load_state_dict(torch.load(io.BytesIO(parameters)))
        model.to(self.device)
        model.eval()

        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, targets in self.test_data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
                predicted = torch.round(outputs).squeeze()
                test_total += targets.size(0)
                test_correct += (predicted == targets.squeeze()).sum().item()

        accuracy = test_correct / test_total
        return accuracy

    def aggregate(self, parameters_list) -> bytes:
        parameters_list = [
            torch.load(io.BytesIO(parameters)) for parameters in parameters_list
        ]
        averaged_parames_template = parameters_list[0]

        for k in averaged_parames_template.keys():
            temp_w = []
            for local_w in parameters_list:
                temp_w.append(local_w[k])
            averaged_parames_template[k] = sum(temp_w) / len(temp_w)

        buffer = io.BytesIO()
        torch.save(averaged_parames_template, buffer)
        aggregated_parameters = buffer.getvalue()
        return aggregated_parameters


if __name__ == "__main__":
    epochs = 100
    lr = 0.000001
    features = 30
    dataset_path = "data/creditCard.json"

    model = ExampleTorchModal(features, dataset_path, epochs=epochs, lr=lr)

    # Training the model
    trained_parameters = model.train()

    # Evaluating the model
    accuracy = model.evaluate(trained_parameters)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    # For aggregation example, you need a list of parameters from different training sessions
    # Since we don't have multiple sessions, we'll simulate it with the same parameters
    parameters_list = [trained_parameters, trained_parameters]  # Example list
    aggregated_parameters = model.aggregate(parameters_list)

    # Re-evaluate the aggregated parameters
    aggregated_accuracy = model.evaluate(aggregated_parameters)
    print(f"Aggregated model accuracy: {aggregated_accuracy * 100:.2f}%")
