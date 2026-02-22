import pydantic
import yaml

from .base import EnvYaml

class DatasetYaml(pydantic.BaseModel):
    name: str
    text_encoder_name: str

class DDEYaml(pydantic.BaseModel):
    num_rounds: int
    num_reverse_rounds: int

class RetrieverYaml(pydantic.BaseModel):
    topic_pe: bool
    DDE_kwargs: DDEYaml
    hidden_dim: int = 256


class MotifYaml(pydantic.BaseModel):
    enabled: bool = True
    backend: str = 'python'
    orca_path: str = ''
    top_k_tokens: int = 4
    motif_emb_dim: int = 64
    vocab_size: int = 17
    query_cross_attn_enabled: bool = False
    motif_residual_blend_enabled: bool = False
    motif_residual_init_alpha: float = 0.2


class LossYaml(pydantic.BaseModel):
    motif_kl_weight: float = 0.1

class OptimizerYaml(pydantic.BaseModel):
    lr: float

class EvalYaml(pydantic.BaseModel):
    k_list: str

class RetrieverExpYaml(pydantic.BaseModel):
    num_epochs: int
    patience: int
    save_prefix: str

class RetrieverTrainYaml(pydantic.BaseModel):
    env: EnvYaml
    dataset: DatasetYaml
    retriever: RetrieverYaml
    motif: MotifYaml = MotifYaml()
    loss: LossYaml = LossYaml()
    optimizer: OptimizerYaml
    eval: EvalYaml
    train: RetrieverExpYaml

def load_yaml(config_file):
    with open(config_file) as f:
        yaml_data = yaml.load(f, Loader=yaml.loader.SafeLoader)

    task = yaml_data.pop('task')
    assert task == 'retriever'
    
    config = RetrieverTrainYaml(**yaml_data).model_dump()
    config['eval']['k_list'] = [
        int(k) for k in config['eval']['k_list'].split(',')]

    return config
