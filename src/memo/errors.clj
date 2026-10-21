(ns memo.errors
  "memo's errors: `ex-info` carrying Python memo's message, a hint, the source
  location and the frame context (DESIGN.md §12). The message content follows
  Python memo 1.3.3's, so compile-error tests compare messages with its own.")

(def ^:dynamic *warnings?*
  "When true (the default), memo's compile-time warnings print to stderr, as
  Python memo's do. Tests bind it to false."
  true)

(defn memo-error
  "A memo error (not thrown). opts: :hint, :user (false marks a suspected bug
  in memo.clj, Python's `user=False`), :loc {:line n :name memo-name},
  :frame-ctxt (a description of the frame the error was found in), :cause
  (the exception behind it)."
  [message {:keys [hint user loc frame-ctxt cause] :or {user true}}]
  (ex-info message
           {:memo/error true
            :memo/hint hint
            :memo/user user
            :memo/loc loc
            :memo/frame-ctxt frame-ctxt}
           cause))

(defn raise
  "Throw a memo error (memo-error's message and opts)."
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
          {:memo/keys [hint user loc frame-ctxt]} (ex-data e)]
      (str "memo error: " (ex-message e)
         (when loc
           (str "\n  in: " (when (:name loc) (str "defmemo " (:name loc)))
                (when (:line loc) (str ", line " (:line loc)))))
         (when hint (str "\n  hint: " hint))
         (when frame-ctxt (str "\n  ctxt: " frame-ctxt))
         (when (false? user)
           "\n  [This may be a bug in memo.clj: please report it.]")))))

(defn warn
  "Print parts as a warning to *err*, unless *warnings?* is false."
  [& parts]
  (when *warnings?*
    (binding [*out* *err*]
      (println (apply str "Warning: " parts)))))

(defn warn-at
  "warn, naming where: loc as a memo error's (:memo/loc), {:name :line}."
  [loc & parts]
  (apply warn (concat parts (when (:name loc)
                              ["\n  in: defmemo " (:name loc)
                               (when (:line loc) (str ", line " (:line loc)))]))))
