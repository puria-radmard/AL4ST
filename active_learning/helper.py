from active_learning.acquisition import RandomBaselineAcquisition, LowestConfidenceAcquisition, \
    MaximumEntropyAcquisition, BALDAcquisition
from active_learning.agent import ActiveLearningAgent
from active_learning.selector import WordWindowSelector, FullSentenceSelector


def configure_al_agent(args, device, model, train_set, helper):
    round_size = int(args.roundsize)

    if args.window != -1:
        selector = WordWindowSelector(window_size=args.window)
    else:
        selector = FullSentenceSelector(helper)

    if args.acquisition == 'baseline' and args.initprop != 1.0:
        raise ValueError("To run baseline, you must set initprop == 1.0")

    if args.acquisition == 'rand' or args.acquisition == 'baseline':
        acquisition_class = RandomBaselineAcquisition()
    elif args.acquisition == 'lc':
        acquisition_class = LowestConfidenceAcquisition()
    elif args.acquisition == 'maxent':
        acquisition_class = MaximumEntropyAcquisition()
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
        model=model,
        device=device
    )

    return agent
