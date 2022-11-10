''' Register rule-based models or pre-trianed models
'''

from rlcard.models.registration import register, load
'''
register(
    model_id = 'leduc-holdem-cfr',
    entry_point='rlcard.models.pretrained_models:LeducHoldemCFRModel')

register(
    model_id = 'leduc-holdem-rule-v1',
    entry_point='rlcard.models.leducholdem_rule_models:LeducHoldemRuleModelV1')

register(
    model_id = 'leduc-holdem-rule-v2',
    entry_point='rlcard.models.leducholdem_rule_models:LeducHoldemRuleModelV2')

register(
    model_id = 'uno-rule-v1',
    entry_point='rlcard.models.uno_rule_models:UNORuleModelV1')

register(
    model_id = 'limit-holdem-rule-v1',
    entry_point='rlcard.models.limitholdem_rule_models:LimitholdemRuleModelV1')

register(
    model_id = 'doudizhu-rule-v1',
    entry_point='rlcard.models.doudizhu_rule_models:DouDizhuRuleModelV1')

register(
    model_id='gin-rummy-novice-rule',
    entry_point='rlcard.models.gin_rummy_rule_models:GinRummyNoviceRuleModel')
'''
register(
    model_id = 'red_10_agent',
    entry_point='rlcard.models.red_10:Red_10RuleModelV1')
register(
    model_id = 'douzero_agent',
    entry_point='rlcard.models.red_10:Red_10RuleModelV2')
register(
    model_id = 'red_10-no_danger_agent',
    entry_point='rlcard.models.red_10:Red_10RuleModelV3')
register(
    model_id = 'CDQN_agent',
    entry_point='rlcard.models.CDQN_agent.CDQN:CDQNModel')
register(
    model_id = 'mfrl_agent',
    entry_point='rlcard.models.mfrl_agent.mfrl:MFRLModel')

register(
    model_id = 'RHCP_agent',
    entry_point='rlcard.models.red_10:RHCPRuleModel')

register(
    model_id = 'Random_agent',
    entry_point='rlcard.models.red_10:RandomModel')