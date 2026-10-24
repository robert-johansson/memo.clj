(ns memo.errors
  "memo's errors: `ex-info` carrying a message, a hint, the source location
  and the frame context. Where Python memo has the error, the message is
  its own (Python memo 1.3.3's), so compile-error tests compare messages
  with it; errors only memo.clj has (the bridge, the loud errors where
  Python memo gives a silent number) have memo.clj's own.

  Stable: memo-error? and root-memo-error recognize a memo error; its
  ex-data keys :memo/error (true), :memo/hint (a string or nil), :memo/loc
  ({:name memo-name :line n}, or nil) and :memo/frame-ctxt. Messages and
  format-error's text are for reading: they may change in any release.")

(def ^:dynamic *warnings?*
  "When true (the default), memo's warnings print to stderr, as Python
  memo's do: when a defmemo is evaluated (a redundant expectation), and at
  a call. Bind it to false to silence them."
  true)

(defn ^:no-doc memo-error
  "Internal: a memo error (not thrown). opts: :hint, :user (false marks a suspected bug
  in memo.clj, Python's `user=False`), :loc {:line n :name memo-name},
  :frame-ctxt (a description of the frame the error was found in), :cause
  (the exception behind it)."
  [message {:keys [hint user loc frame-ctxt cause] :or {user true}}]
  (ex-info message
           {:memo/error true
            :memo/hint hint
            :memo.impl/user user
            :memo/loc loc
            :memo/frame-ctxt frame-ctxt}
           cause))

(defn ^:no-doc raise
  "Internal: throw a memo error (memo-error's message and opts)."
  ([message] (raise message {}))
  ([message opts] (throw (memo-error message opts))))

(defn memo-error?
  "Whether e is a memo error itself (see root-memo-error for one Clojure
  wrapped)."
  [e]
  (boolean (some-> (ex-data e) :memo/error)))

(defn root-memo-error
  "The memo error inside e, which Clojure may have wrapped (a defmemo that fails
  to compile throws from macroexpansion or from its def), or nil."
  [e]
  (loop [e e]
    (cond
      (nil? e) nil
      (memo-error? e) e
      :else (recur (ex-cause e)))))

(defn format-error
  "e rendered as Python memo renders its notes: the memo error inside it,
  when Clojure wrapped one (as a defmemo that fails to compile does). Any
  other exception is named by its class, as not a memo error."
  [e]
  (if-not (root-memo-error e)
    (str "error (not a memo error): "
         (if e (str (.getSimpleName (class e)) ": " (ex-message e)) "nil"))
    (let [e (root-memo-error e)
          {:memo/keys [hint loc frame-ctxt] user :memo.impl/user} (ex-data e)]
      (str "memo error: " (ex-message e)
         (when loc
           (str "\n  in: " (when (:name loc) (str "defmemo " (:name loc)))
                (when (:line loc) (str ", line " (:line loc)))))
         (when hint (str "\n  hint: " hint))
         (when frame-ctxt (str "\n  ctxt: " frame-ctxt))
         (when (false? user)
           "\n  [This may be a bug in memo.clj: please report it.]")))))

(defn ^:no-doc warn
  "Internal: print parts as a warning to *err*, unless *warnings?* is false."
  [& parts]
  (when *warnings?*
    (binding [*out* *err*]
      (println (apply str "Warning: " parts)))))

(defn ^:no-doc warn-at
  "Internal: warn, naming where: loc as a memo error's (:memo/loc), {:name :line}."
  [loc & parts]
  (apply warn (concat parts (when (:name loc)
                              ["\n  in: defmemo " (:name loc)
                               (when (:line loc) (str ", line " (:line loc)))]))))
