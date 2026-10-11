import functools

def log_step(description: str = None):
    """
    Decorate a step method so it logs success or failure and stops the pipeline on error.

    The decorated method's class must set `self.logger`.

    Args:
        description (str): Short description of the step, used in log messages.

    Returns:
        Callable: Decorator for the step method.
    """

    def decorator(method):
        """
        Wrap the step method.

        Args:
            method (Callable): Step method to wrap.

        Returns:
            Callable: Wrapped step method.
        """

        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            """
            Run the step and log the outcome.

            Args:
                self (object): Instance the step belongs to.
                *args (tuple): Positional arguments for the step.
                **kwargs (dict): Keyword arguments for the step.

            Returns:
                Any: The step's return value.
            """

            try:

                # Run the step.
                result = method(self, *args, **kwargs)

            except Exception as e:

                # Log error under the step's name.
                self.logger.error(f"{description} failed: {e}.", extra = {"step": method.__name__})

                # Exit.
                raise SystemExit from e

            # Log success under the step's name.
            self.logger.info(f"{description}.", extra = {"step": method.__name__})

            # Return step result.
            return result

        # Return wrapped method.
        return wrapper

    # Return decorator.
    return decorator
