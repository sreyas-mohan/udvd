from collections import OrderedDict
from numbers import Number
from tqdm import tqdm
from .meters import AverageMeter, RunningAverageMeter, TimeMeter


class ProgressBar:
    def __init__(self, iterable, epoch=None, prefix=None, quiet=False):
        self.epoch = epoch
        self.quiet = quiet
        self.prefix = prefix + ' | ' if prefix is not None else ''
        if epoch is not None:
            self.prefix += f"epoch {epoch:02d}"
        self.iterable = iterable if self.quiet else tqdm(iterable, self.prefix, leave=False)

    def __iter__(self):
        return iter(self.iterable)

    def log(self, stats, verbose=False):
        if not self.quiet:
            self.iterable.set_postfix(self.format_stats(stats, verbose), refresh=True)

    def format_stats(self, stats, verbose=False):
        postfix = OrderedDict(stats)
        for key, value in postfix.items():
            if isinstance(value, Number):
                fmt = "{:.3f}" if value > 0.001 else "{:.1e}"
                postfix[key] = fmt.format(value)
            elif isinstance(value, AverageMeter) or isinstance(value, RunningAverageMeter):
                if verbose:
                    postfix[key] = f"{value.avg:.3f} ({value.val:.3f})"
                else:
                    postfix[key] = f"{value.avg:.3f}"
            elif isinstance(value, TimeMeter):
                postfix[key] = f"{value.elapsed_time:.1f}s"
            elif not isinstance(postfix[key], str):
                postfix[key] = str(value)
        return postfix

    def print(self, stats, verbose=False):
        postfix = " | ".join(key + " " + value.strip() for key, value in self.format_stats(stats, verbose).items())
        return f"{self.prefix + ' | ' if self.epoch is not None else ''}{postfix}"
