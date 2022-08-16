

from ogb.linkproppred import Evaluator



class InteractionEvaluator:

    def __init__(self, dataset_configs):
        
        self.dataset_configs = dataset_configs
        self.dataset_name = dataset_configs['name']


        if self.dataset_name[:4] == 'ogbl':
            self.evaluator = Evaluator(self.dataset_name.replace('_', '-'))
        
        else:
            self.evaluator = self.build_evaluator()

    
    def build_evaluator(self):
        return None
