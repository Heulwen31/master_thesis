from river import forest, tree, ensemble

class RiverModelFactory:
    @staticmethod
    def get_model(algorithm, **kwargs):
        """
        Returns an initialized River model instance.
        """
        n_models = kwargs.get("n_models", 5)
        lambda_value = kwargs.get("lambda_value", 6)
        grace_period = kwargs.get("grace_period", 50)
        split_criterion = kwargs.get("split_criterion", "info_gain")
        seed = kwargs.get("seed", 42)

        if algorithm == "ARF":
            return forest.ARFClassifier(
                n_models=n_models,
                lambda_value=lambda_value,
                grace_period=grace_period,
                split_criterion=split_criterion,
                seed=seed,
            )
        elif algorithm == "HAT":
            return tree.HoeffdingAdaptiveTreeClassifier(
                seed=seed
            )
        elif algorithm == "LeveragingBagging":
            return ensemble.LeveragingBaggingClassifier(
                model=tree.HoeffdingTreeClassifier(split_criterion=split_criterion),
                n_models=n_models,
                seed=seed
            )
        else:
            raise ValueError(f"Unknown River algorithm: {algorithm}")
