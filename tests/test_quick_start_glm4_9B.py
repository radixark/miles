import command_utils as U


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command("hf download zai-org/GLM-Z1-9B-0414 --local-dir /root/models/GLM-Z1-9B-0414")
    U.exec_command("hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k")
    U.exec_command("hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/datasets/aime-2024")

    U.convert_checkpoint(model_name="GLM-Z1-9B-0414", model_type="glm4-9B")


def execute():
    U.ray_start()


if __name__ == '__main__':
    pass  # TODO make the functions executable via typer
