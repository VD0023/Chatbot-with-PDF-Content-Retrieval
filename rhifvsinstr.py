#Rlhf vs InstTuning
class RLHFvsInstructionFineTuning:
    def __init__(self):
        self.rlhf_model = None
        self.instruction_fine_tuning_model = None

    def train_rlhf_model(self, feedback):
        pass

    def train_instruction_fine_tuning_model(self, instructions):
        pass

"""

Reinforcement Learning from Human Feedback (RLHF) involves training a model by receiving feedback from humans,
typically in the form of comparisons or rankings. Instruction fine-tuning, on the other hand, relies on explicit
guidance or instructions during training. RLHF is more interactive, learning from feedback loops,
while instruction fine-tuning is based on predefined instructions.

"""