class Log:
    def __init__(self, config: Config, objective: Objective):
        self._config = config
        self._objective = objective
        self.out_path = config.out_path
        self.loss_log, self.gradient_log, self.parameters_log = [], [], []
        self.prepare_path()

    def prepare_path(self):
        os.makedirs(f"{self.out_path}/log", exist_ok=True)
        log_num = len(os.listdir(f"{self.out_path}/log")) + 1
        self._log_path = f"{self.out_path}/log/{log_num}"
        os.makedirs(self._log_path, exist_ok=True)

    def save(self):
        self.loss_log.append(self._objective.loss.detach().cpu().tolist())
        self.gradient_log.append(self._objective.gradient.detach().cpu().tolist())
        self.parameters_log.append(self._objective.parameters.detach().cpu().tolist())
        self._config.to_yaml(path=self._log_path, name="config")
        torch.save(torch.tensor(self.loss_log), f"{self._log_path}/loss.pt")
        torch.save(torch.tensor(self.gradient_log), f"{self._log_path}/gradient.pt")
        torch.save(torch.tensor(self.parameters_log), f"{self._log_path}/parameter.pt")

    @staticmethod
    def plot(path):
        loss = torch.load(f"{path}/loss.pt")
        gradient = torch.load(f"{path}/gradient.pt")
        parameter = torch.load(f"{path}/parameter.pt")
        for tensor, label in zip([loss, gradient, parameter], ["Loss", "Gradient", "Parameter"]):
            plt.figure()
            plt.plot(tensor)
            plt.xlabel("# Iterations")
            plt.ylabel(label)
        plt.show()
