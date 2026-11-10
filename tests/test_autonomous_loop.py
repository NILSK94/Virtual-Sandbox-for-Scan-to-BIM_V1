"""Offline regression coverage for autonomous loop recovery."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
from scanbim import agents
from scanbim.model import SceneStore
from scanbim.settings import Credentials


class AutonomousLoopTests(unittest.TestCase):
    def run_actions(self, actions):
        with tempfile.TemporaryDirectory() as folder:
            scene = SceneStore(Path(folder)).create(
                np.array([[0., 0., 0.], [1., 1., 1.], [2., 1., 2.]]),
                np.full((3, 3), 120, dtype=np.uint8), 'probe', 'hash')
            scene.state['experiments'].append({'id':'source', 'name':'Measurements',
                'ok':True, 'code':'result={}', 'result':{'height':3}})
            requests=[]
            def handle(request):
                body=json.loads(request.content);requests.append(copy.deepcopy(body))
                action=actions[len(requests)-1]
                return httpx.Response(200,json={'output':[{'type':'function_call','name':'act',
                    'call_id':str(len(requests)), 'arguments':json.dumps(action)}]})
            real=httpx.Client
            with patch.object(agents, 'load_credentials', return_value=Credentials('test', 'test', 'openai')), \
                 patch.object(agents.httpx, 'Client', lambda **kw: real(transport=httpx.MockTransport(handle), **kw)):
                summary=agents.AgentRunner()._vlm(scene,len(actions)+2)
            return summary,requests

    @staticmethod
    def action(tool,args):
        return {'tool':tool,'arguments_json':json.dumps(args),'rationale':'Regression probe'}

    def test_long_retrieval_does_not_abort_and_receives_build_feedback(self):
        actions=[self.action('read_experiment',{'experiment_id':'source'}) for _ in range(9)]
        actions.append(self.action('finish',{'status':'partial','summary':'Evidence retrieved'}))
        summary,requests=self.run_actions(actions)
        self.assertEqual(summary,'Evidence retrieved')
        self.assertEqual(len(requests),10)
        self.assertIn('Measured support should become',json.dumps(requests[7]))

    def test_malformed_first_arguments_are_recoverable(self):
        bad={'tool':'run_experiment','arguments_json':'{broken','rationale':'Probe'}
        summary,requests=self.run_actions([bad,self.action('finish',{'status':'partial','summary':'Recovered'})])
        self.assertEqual(summary,'Recovered')
        self.assertIn('error',json.dumps(requests[1]))

    def test_repeated_action_yields_feedback_instead_of_stopping(self):
        actions=[self.action('request_view',{'label':'same'}) for _ in range(3)]
        actions.append(self.action('finish',{'status':'partial','summary':'Changed approach'}))
        summary,requests=self.run_actions(actions)
        self.assertEqual(summary,'Changed approach')
        self.assertIn('Unchanged action already attempted',json.dumps(requests[-1]))

if __name__=='__main__':
    unittest.main()
