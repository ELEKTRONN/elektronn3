import code
import getpass
import time
import traceback
import signal
import prompt_toolkit
from  prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.contrib.completers import PathCompleter
from prompt_toolkit.contrib.regular_languages.compiler import \
    compile as compile_grammar
from prompt_toolkit.contrib.regular_languages.completion import \
    GrammarCompleter
from torch.utils.data.dataloader import DataLoader, DataLoaderIter
from ..data.utils import DelayedInterrupt, GracefulInterrupt
import re
import gc
import jedi
import numpy as np
from ..data.utils import pickleload, picklesave
from . import plotting
from .. import floatX
import matplotlib.pyplot as plt
shortcut_completions = [  # Extra words to register completions for:
    'q', 'kill', 'sethist', 'setlr', 'setmom', 'setwd', 'sf', 'preview',
    'paramstats', 'gradstats', 'actstats', 'debugbatch', 'load']
user_name = getpass.getuser()
ptk_hist = InMemoryHistory()


def user_input(local_vars):
    _banner = """
    ========================
    === ELEKTRONN3 SHELL ===
    ========================
    >>>>>>>>>>>>>>>>> UNDER DEVELOPMENT <<<<<<<<<<<<<<<<<
    Shortcuts:
    'help' (display this help text),
    'q' (leave menu),         'kill'(saving last params),
    'ip' (start embedded IPython shell)

    For everything else enter a command in the command line\n"""

    _ipython_banner = """    You are now in the embedded IPython shell.
    You still have full access to the local scope of the ELEKTRONN2 shell
    (e.g. 'neural', ), but shortcuts like 'q' no longer work.

    To leave the IPython shell and switch back to the ELEKTRONN2 shell, run
    'exit()' or hit 'Ctrl-D'."""
    gc.collect()
    print(_banner)
    trainer = local_vars['self']
    data = trainer.dataset
    model = trainer.model
    loss = trainer.criterion
    optimizer = trainer.optimizer
    local_vars.update(locals())  # put the above into scope of console
    console = code.InteractiveConsole(locals=local_vars)

    while True:
        try:
            try:
                inp = prompt_toolkit.prompt(u"%s@neuromancer: " % user_name,
                                            # needs to be an explicit ustring for py2-compat
                                            history=ptk_hist,
                                            completer=NumaCompleter(
                                                lambda: local_vars, lambda: {},
                                                words=shortcut_completions,
                                                words_metastring='(shortcut)'),
                                            auto_suggest=AutoSuggestFromHistory())
            # Catch all exceptions in order to prevent catastrophes in case ptk suddenly breaks
            except Exception:
                inp = console.raw_input("%s@neuromancer: " % user_name)
            if inp=='q':
                break
            elif inp=='kill':
                break
            elif inp == 'help':
                print(_banner)
            elif inp == 'ip':
                try:
                    import IPython
                    IPython.embed(header=_ipython_banner)
                except ImportError:
                    print('IPython is not available. You will need to install '
                          'it to use this function.')
            else:
                console.push(inp)

        except KeyboardInterrupt:
            print('Enter "q" to leave the shell and continue training.\n'
                  'Enter "kill" to kill the training, saving current parameters.')
        except IndexError as err:
            if any([inp.startswith(shortcut) for shortcut in shortcut_completions]):  # ignore trailing spaces
                print('IndexError. Probably you forgot to type a value after the shortcut "{}".'.format(inp))
            else:
                raise err  # All other IndexErrors are already correctly handled by the console.
        except ValueError as err:
            if any([inp.startswith(shortcut) for shortcut in shortcut_completions]):  # ignore trailing spaces
                print('ValueError. The "{}" shortcut received an unexpected argument.'.format(inp))
            else:
                raise err  # All other IndexErrors are already correctly handled by the console.
        except Exception:
            traceback.print_exc()
            print('\n\nUnhandled exception occured. See above traceback for debug info.\n'
                  'If you think this is a bug, please consider reporting it at '
                  'https://github.com/ELEKTRONN/ELEKTRONN2/issues.')

    return inp


# Based on https://github.com/jonathanslenders/ptpython/blob/master/ptpython/completer.py,
#     with additional word completions through the words argument
class NumaCompleter(Completer):
    """
    Completer for Python, file system paths and custom words
    """

    def __init__(self, get_globals, get_locals, words=None,
                 words_metastring=''):
        super(NumaCompleter, self).__init__()

        if words is None:
            words = []

        self.get_globals = get_globals
        self.get_locals = get_locals
        self.words = words

        # Appears next to all word completions to distinguish them from the Python language completions
        self.words_metastring = words_metastring

        self._path_completer_cache = None
        self._path_completer_grammar_cache = None

    @property
    def _path_completer(self):
        if self._path_completer_cache is None:
            self._path_completer_cache = GrammarCompleter(
                self._path_completer_grammar,
                {'var1': PathCompleter(expanduser=True),
                 'var2': PathCompleter(expanduser=True),})
        return self._path_completer_cache

    @property
    def _path_completer_grammar(self):
        """
        Return the grammar for matching paths inside strings inside Python
        code.
        """
        # We make this lazy, because it delays startup time a little bit.
        # This way, the grammar is build during the first completion.
        if self._path_completer_grammar_cache is None:
            self._path_completer_grammar_cache = self._create_path_completer_grammar()
        return self._path_completer_grammar_cache

    def _create_path_completer_grammar(self):
        def unwrapper(text):
            return re.sub(r'\\(.)', r'\1', text)

        def single_quoted_wrapper(text):
            return text.replace('\\', '\\\\').replace("'", "\\'")

        def double_quoted_wrapper(text):
            return text.replace('\\', '\\\\').replace('"', '\\"')

        grammar = r"""
                # Text before the current string.
                (
                    [^'"#]                                  |  # Not quoted characters.
                    '''  ([^'\\]|'(?!')|''(?!')|\\.])*  ''' |  # Inside single quoted triple strings
                    "" " ([^"\\]|"(?!")|""(?!^)|\\.])* "" " |  # Inside double quoted triple strings

                    \#[^\n]*(\n|$)           |  # Comment.
                    "(?!"") ([^"\\]|\\.)*"   |  # Inside double quoted strings.
                    '(?!'') ([^'\\]|\\.)*'      # Inside single quoted strings.

                        # Warning: The negative lookahead in the above two
                        #          statements is important. If we drop that,
                        #          then the regex will try to interpret every
                        #          triple quoted string also as a single quoted
                        #          string, making this exponentially expensive to
                        #          execute!
                )*
                # The current string that we're completing.
                (
                    ' (?P<var1>([^\n'\\]|\\.)*) |  # Inside a single quoted string.
                    " (?P<var2>([^\n"\\]|\\.)*)    # Inside a double quoted string.
                )
        """

        return compile_grammar(grammar,
                               escape_funcs={'var1': single_quoted_wrapper,
                                             'var2': double_quoted_wrapper,},
                               unescape_funcs={'var1': unwrapper,
                                               'var2': unwrapper,})

    def _complete_path_while_typing(self, document):
        char_before_cursor = document.char_before_cursor
        return document.text and (
            char_before_cursor.isalnum() or char_before_cursor in '/.~')

    def _complete_python_while_typing(self, document):
        char_before_cursor = document.char_before_cursor
        return document.text and (
            char_before_cursor.isalnum() or char_before_cursor in '_.')

    def get_completions(self, document, complete_event):
        """
        Get completions.
        """

        # Do Path completions
        if complete_event.completion_requested or self._complete_path_while_typing(
                document):
            for c in self._path_completer.get_completions(document,
                                                          complete_event):
                yield c

        # If we are inside a string, Don't do Jedi completion.
        if self._path_completer_grammar.match(document.text_before_cursor):
            return

        # Do custom word completions (only if the word is at the beginning of the line)
        if complete_event.completion_requested or self._complete_python_while_typing(
                document):
            for word in self.words:
                line_before_cursor = document.current_line_before_cursor
                if word.startswith(line_before_cursor):
                    yield Completion(word, -len(line_before_cursor),
                                     display_meta=self.words_metastring)

        # Do Jedi Python completions.
        if complete_event.completion_requested or self._complete_python_while_typing(
                document):
            script = _get_jedi_script_from_document(document, self.get_locals(),
                                                    self.get_globals())

            if script:
                try:
                    completions = script.completions()
                except TypeError:
                    # Issue #9: bad syntax causes completions() to fail in jedi.
                    # https://github.com/jonathanslenders/python-prompt-toolkit/issues/9
                    pass
                except UnicodeDecodeError:
                    # Issue #43: UnicodeDecodeError on OpenBSD
                    # https://github.com/jonathanslenders/python-prompt-toolkit/issues/43
                    pass
                except AttributeError:
                    # Jedi issue #513: https://github.com/davidhalter/jedi/issues/513
                    pass
                except ValueError:
                    # Jedi issue: "ValueError: invalid \x escape"
                    pass
                except KeyError:
                    # Jedi issue: "KeyError: u'a_lambda'."
                    # https://github.com/jonathanslenders/ptpython/issues/89
                    pass
                except IOError:
                    # Jedi issue: "IOError: No such file or directory."
                    # https://github.com/jonathanslenders/ptpython/issues/71
                    pass
                else:
                    for c in completions:
                        yield Completion(c.name_with_symbols,
                                         len(c.complete) - len(
                                             c.name_with_symbols),
                                         display=c.name_with_symbols)


# From https://github.com/jonathanslenders/ptpython/blob/master/ptpython/utils.py
def _get_jedi_script_from_document(document, locals, globals):
    try:
        return jedi.Interpreter(document.text,
                                column=document.cursor_position_col,
                                line=document.cursor_position_row + 1,
                                path='input-text',
                                namespaces=[locals, globals])
    except ValueError:
        # Invalid cursor position.
        # ValueError('`column` parameter is not in a valid range.')
        return None
    except AttributeError:
        # Workaround for #65: https://github.com/jonathanslenders/python-prompt-toolkit/issues/65
        # See also: https://github.com/davidhalter/jedi/issues/508
        return None
    except IndexError:
        # Workaround Jedi issue #514: for https://github.com/davidhalter/jedi/issues/514
        return None
    except KeyError:
        # Workaroud for a crash when the input is "u'", the start of a unicode string.
        return None


class HistoryTracker(object):
    def __init__(self):
        self.plotting_proc = None
        self.debug_outputs = None
        self.regression_track = None
        self.debug_output_names = None

        self.timeline = AccumulationArray(n_init=int(1e5), dtype=dict(
            names=[u'time', u'loss', u'batch_char', ], formats=[u'f4', ] * 3))

        self.history = AccumulationArray(n_init=int(1e4), dtype=dict(
            names=[u'steps', u'time', u'train_loss', u'valid_loss',
                   u'loss_gain', u'train_err', u'valid_err', u'lr', u'mom',
                   u'gradnetrate'], formats=[u'i4', ] + [u'f4', ] * 9))
        self.loss = AccumulationArray(n_init=int(1e5), data=[])

    def update_timeline(self, vals):
        self.timeline.append(vals)
        self.loss.append(vals[1])

    def register_debug_output_names(self, names):
        self.debug_output_names = names

    def update_history(self, vals):
        self.history.append(vals)

    def update_debug_outputs(self, vals):
        if self.debug_outputs is None:
            self.debug_outputs = AccumulationArray(n_init=int(1e5),
                                                   right_shape=len(vals))

        self.debug_outputs.append(vals)

    def update_regression(self, pred, target):
        if self.regression_track is None:
            assert len(pred)==len(target)
            p = AccumulationArray(n_init=int(1e5), right_shape=len(pred))
            t = AccumulationArray(n_init=int(1e5), right_shape=len(pred))
            self.regression_track = [p, t]

        self.regression_track[0].append(pred)
        self.regression_track[1].append(target)

    def save(self, save_name):
        file_name = save_name + '.history.pkl'
        picklesave([self.timeline, self.history, self.debug_outputs,
                          self.debug_output_names, self.regression_track],
                         file_name)

    def load(self, file_name):
        (self.timeline, self.history, self.debug_outputs,
         self.debug_output_names, self.regression_track) = pickleload(
            file_name)

    def plot(self, save_name=None, autoscale=True, close=True, loss_smoothing_len=200):
        plotting.plot_hist(self.timeline, self.history, save_name,
                           loss_smoothing_len, autoscale)

        if self.debug_output_names and self.debug_outputs.length:
            plotting.plot_debug(self.debug_outputs, self.debug_output_names,
                                save_name)

        if self.regression_track:
            plotting.plot_regression(self.regression_track[0],
                                     self.regression_track[1], save_name)
            plotting.plot_kde(self.regression_track[0],
                              self.regression_track[1], save_name)

        if close:
            plt.close('all')


class AccumulationArray(object):
    def __init__(self, right_shape=(), dtype=floatX, n_init=100, data=None,
                 ema_factor=0.95):
        if isinstance(dtype, dict) and right_shape!=():
            raise ValueError("If dict is used as dtype, right shape must be"
                             "unchanged (i.e it is 1d)")

        if data is not None and len(data):
            n_init += len(data)
            right_shape = data.shape[1:]
            dtype = data.dtype

        self._n_init = n_init
        if isinstance(right_shape, int):
            self._right_shape = (right_shape,)
        else:
            self._right_shape = tuple(right_shape)
        self.dtype = dtype
        self.length = 0
        self._buffer = self._alloc(n_init)
        self._min = +np.inf
        self._max = -np.inf
        self._sum = 0
        self._ema = None
        self._ema_factor = ema_factor

        if data is not None and len(data):
            self.length = len(data)
            self._buffer[:self.length] = data
            self._min = data.min(0)
            self._max = data.max(0)
            self._sum = data.sum(0)

    def __repr__(self):
        return repr(self.data)

    def _alloc(self, n):
        if isinstance(self._right_shape, (tuple, list, np.ndarray)):
            ret = np.zeros((n,) + tuple(self._right_shape), dtype=self.dtype)
        elif isinstance(self.dtype, dict):  # rec array
            ret = np.zeros(n, dtype=self.dtype)
        else:
            raise ValueError("dtype not understood")
        return ret

    def append(self, data):
        # data = self.normalise_data(data)
        if len(self._buffer)==self.length:
            tmp = self._alloc(len(self._buffer) * 2)
            tmp[:self.length] = self._buffer
            self._buffer = tmp

        if isinstance(self.dtype, dict):
            for k, val in enumerate(data):
                self._buffer[self.length][k] = data[k]
        else:
            self._buffer[self.length] = data
            if self._ema is None:
                self._ema = self._buffer[self.length]
            else:
                f = self._ema_factor
                fc = 1 - f
                self._ema = self._ema * f + self._buffer[self.length] * fc

        self.length += 1

        self._min = np.minimum(data, self._min)
        self._max = np.maximum(data, self._max)
        self._sum = self._sum + np.asanyarray(data)

    def add_offset(self, off):
        self.data[:] += off
        if off.ndim>np.ndim(self._sum):
            off = off[0]
        self._min += off
        self._max += off
        self._sum += off * self.length

    def clear(self):
        self.length = 0
        self._min = +np.inf
        self._max = -np.inf
        self._sum = 0

    def mean(self):
        return np.asarray(self._sum, dtype=floatX) / self.length

    def sum(self):
        return self._sum

    def max(self):
        return self._max

    def min(self):
        return self._min

    def __len__(self):
        return self.length

    @property
    def data(self):
        return self._buffer[:self.length]

    @property
    def ema(self):
        return self._ema

    def __getitem__(self, slc):
        return self._buffer[:self.length][slc]


class Timer(object):
    def __init__(self):
        self.origin = time.time()
        self.t0 = self.origin

    @property
    def t_passed(self):
        return time.time() - self.origin


def pretty_string_time(t):
    """Custom printing of elapsed time"""
    if t > 4000:
        s = 't=%.1fh' % (t / 3600)
    elif t > 300:
        s = 't=%.0fm' % (t / 60)
    else:
        s = 't=%.0fs' % (t)
    return s


class DelayedDataLoaderIter(DataLoaderIter):
    def __init__(self, loader):
        try:
            with DelayedInterrupt([signal.SIGTERM, signal.SIGINT]):
                super(DelayedDataLoaderIter, self).__init__(loader)
        except KeyboardInterrupt:
            self.shutdown = True
            self._shutdown_workers()
            for w in self.workers:
                w.terminate()
            raise KeyboardInterrupt

    def __next__(self):
        try:
            with DelayedInterrupt([signal.SIGTERM, signal.SIGINT]):
                nxt = super(DelayedDataLoaderIter, self).__next__()
            return nxt
        except KeyboardInterrupt:
            self.shutdown = True
            self._shutdown_workers()
            for w in self.workers:
                w.terminate()
            raise KeyboardInterrupt


class DelayedDataLoader(DataLoader):
    def __iter__(self):
        return DelayedDataLoaderIter(self)
