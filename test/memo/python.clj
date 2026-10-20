(ns memo.python
  "Python memo, run from Clojure: the oracle as optional tooling (DESIGN.md
  §11), the pinned memo in memo-mcp/memo with memo-mcp's venv. Tests that
  need it check available? and skip loudly without it.

  run takes a module's source (memo.pygen's, guarded) and calls of its
  models, runs them all in one Python process, and returns each call's
  outcome: {:outcome :ok :shape [..] :value nested-vectors}, {:outcome :ce
  :message} (the model did not compile) or {:outcome :re :message} (the
  call failed)."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.java.shell :as sh]
            [clojure.string :as str]))

(def venv-python "memo-mcp/venv/bin/python")
(def memo-src "memo-mcp/memo")

(defn available? []
  (and (.exists (io/file venv-python)) (.exists (io/file memo-src "memo" "core.py"))))

(defn- json
  "Clojure data as JSON: numbers, strings, keywords (as strings), booleans,
  nil, vectors and maps."
  [x]
  (cond
    (nil? x) "null"
    (boolean? x) (str x)
    (integer? x) (str x)
    (number? x) (let [d (double x)]
                  (cond (Double/isNaN d) "NaN"
                        (Double/isInfinite d) (if (pos? d) "Infinity" "-Infinity")
                        :else (str d)))
    (string? x) (pr-str x)
    (keyword? x) (pr-str (name x))
    (symbol? x) (pr-str (name x))
    (map? x) (str "{" (str/join ", " (map (fn [[k v]] (str (json (name k)) ": " (json v))) x)) "}")
    (sequential? x) (str "[" (str/join ", " (map json x)) "]")
    :else (throw (ex-info (str "no JSON for " (pr-str x)) {:x x}))))

(def ^:private driver
  "import sys, json, math, runpy, os
sys.path.insert(0, os.path.abspath(sys.argv[1]))
import numpy as onp
module, calls, out = sys.argv[2:5]
sys.setrecursionlimit(20000)  # demo-mdp's Q(1000), as the fixtures' recorder

def edn(x):
    if x is None: return 'nil'
    if isinstance(x, bool): return 'true' if x else 'false'
    if isinstance(x, int): return str(x)
    if isinstance(x, float):
        if math.isnan(x): return '##NaN'
        if math.isinf(x): return '##Inf' if x > 0 else '##-Inf'
        return repr(x)
    if isinstance(x, str): return json.dumps(x)
    if isinstance(x, dict): return '{' + ' '.join(':' + k + ' ' + edn(v) for k, v in x.items()) + '}'
    if isinstance(x, (list, tuple)): return '[' + ' '.join(edn(v) for v in x) + ']'
    raise TypeError(repr(x))

def tolist(a):
    a = onp.asarray(a)
    return float(a) if a.ndim == 0 else [tolist(v) for v in a]

g = runpy.run_path(module, run_name='memo_fuzz')
results = []
for c in json.load(open(calls)):
    name = c['model']
    if name in g['_compile_errors']:
        results.append({'outcome': ':ce', 'message': g['_compile_errors'][name]})
        continue
    f = g.get(name)
    if f is None:
        results.append({'outcome': ':missing'})
        continue
    try:
        r = onp.asarray(f(*c['args'], **c['kwargs']))
        shape = tuple(f._shape) if f._num_retvals == 1 else (f._num_retvals,) + tuple(f._shape)
        results.append({'outcome': ':ok', 'shape': list(shape), 'value': tolist(onp.broadcast_to(r, shape))})
    except Exception as e:
        results.append({'outcome': ':re', 'message': getattr(e, 'message', None) or f'{type(e).__name__}: {e}'})
with open(out, 'w') as o:
    o.write(edn(results))
")

(defn run
  "Run module (Python source) and then calls, each {:model name :args [...]
  :kwargs {...}} (name as Python's), in one Python process. The outcomes,
  in order, as described in the namespace docstring."
  [module calls]
  (let [dir (.toFile (java.nio.file.Files/createTempDirectory "memo-python" (make-array java.nio.file.attribute.FileAttribute 0)))
        f (fn [n] (io/file dir n))]
    (try
      (spit (f "module.py") module)
      (spit (f "calls.json") (json (vec calls)))
      (spit (f "driver.py") driver)
      (let [{:keys [exit err]} (sh/sh venv-python "-I" (str (f "driver.py")) memo-src
                                      (str (f "module.py")) (str (f "calls.json")) (str (f "out.edn")))]
        (when-not (zero? exit)
          (throw (ex-info (str "Python memo failed: " (last (str/split-lines err))) {:err err})))
        (mapv (fn [r] (update r :outcome #(keyword (subs % 1))))
              (edn/read-string (slurp (f "out.edn")))))
      (finally
        (doseq [x (reverse (file-seq dir))] (.delete ^java.io.File x))))))
