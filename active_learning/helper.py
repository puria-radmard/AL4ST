from active_learning.acquisition import RandomBaselineAcquisition, LowestConfidenceAcquisition, \
    MaximumEntropyAcquisition, BALDAcquisition
from active_learning.agent import ActiveLearningAgent
from active_learning.selector import WordWindowSelector, SentenceSelector


def configure_al_agent(args, device, model, train_set, helper):

    round_size = int(args.roundsize)

    if args.acquisition in ['lc']:
        average = False
    else:
        average = True

    if args.window != -1:
        selector = WordWindowSelector(helper=helper, average=average, window_size=args.window)
    else:
        selector = SentenceSelector(helper, average=average)

    if args.acquisition == 'baseline' and args.initprop != 1.0:
        raise ValueError("To run baseline, you must set initprop == 1.0")

    if args.acquisition == 'rand' or args.acquisition == 'baseline':
        acquisition_class = RandomBaselineAcquisition(model)
    elif args.acquisition == 'lc' or args.acquisition == 'mnlp':
        acquisition_class = LowestConfidenceAcquisition(model)
    elif args.acquisition == 'maxent':
        acquisition_class = MaximumEntropyAcquisition(model)
    elif args.acquisition == 'bald':
        acquisition_class = BALDAcquisition()
    else:
        raise ValueError(args.acquisition)

    agent = ActiveLearningAgent(
        train_set=train_set,
        acquisition_class=acquisition_class,
        selector_class=selector,
        round_size=round_size,
        batch_size=args.batch_size,
        helper=helper,
        device=device
    )

    return agent
