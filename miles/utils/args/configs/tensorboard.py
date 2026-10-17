from miles.utils.args.schema import A, Arg, BaseConfig


# tensorboard
class TensorboardConfig(BaseConfig):
    # tb_project_name, tb_experiment_name
    use_tensorboard: A[bool, Arg()] = False
    tb_project_name: A[
        str | None,
        Arg(help="Directory to store tensorboard logs. Default is  os.environ.get('TENSORBOARD_DIR') directory."),
    ] = None
    tb_experiment_name: A[str | None, Arg()] = None
