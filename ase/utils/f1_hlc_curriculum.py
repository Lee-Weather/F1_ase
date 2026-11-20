"""Configuration-only F1 HLC command curriculum (epochs are PPO iterations)."""


def command_stage(config, epoch):
    stages = config.get("commandCurriculum", [])
    for index, stage in enumerate(stages):
        if epoch <= stage["untilEpoch"] or index == len(stages)-1:
            return index+1, stage
    return 0, config
