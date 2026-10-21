(ns memo.frames
  "The frame tree (lifted from memo-clj after review). A frame is one agent's
  perspective:

  {:name sym :variant :adult|:child|…
   :choices {[who id] {:tag sym :idx dim :known bool :domain sym}}
   :children {name frame}
   :conditions {[who id] [who id]}   ; aliases into the parent frame
   :goals {id expr}
   :ll sym-or-nil                    ; the frame's belief array
   :ll-dims #{dim}                   ; the dims it varies along
   :scrambled nil-or-info}           ; set when an observation scrambled it

  The tree lives in the analysis state under :root, and :path (child names
  from the root) addresses the current frame. Frames are immutable values, so
  a copy (a snapshot, an imagined world) is the value itself."
  (:require [clojure.string :as str]))

(def root-name 'observer)

(defn frame [name variant]
  {:name name :variant variant
   :choices {} :children {} :conditions {} :goals {} :ll nil :ll-dims #{} :scrambled nil})

(defn frame-path [path]
  (vec (mapcat (fn [k] [:children k]) path)))

(defn get-frame
  ([state] (get-frame state (:path state)))
  ([state path] (get-in (:root state) (frame-path path))))

(defn update-frame* [state path f & args]
  (apply update-in state (into [:root] (frame-path path)) f args))

(defn update-frame [state f & args]
  (apply update-frame* state (:path state) f args))

(defn- next-seq [m] (inc (reduce max -1 (keep :seq (vals m)))))

(defn put-choice
  "Frame f with choice ch at key k, numbered (:seq) in the order keys were
  first put, as Python memo's dicts keep them (INSPECT lists them so)."
  [f k ch]
  (assoc-in f [:choices k] (assoc ch :seq (or (get-in f [:choices k :seq]) (next-seq (:choices f))))))

(defn ensure-child
  "The state, with a child named who under the current frame (inheriting its
  variant), numbered (:seq) in the order children were made."
  [state who]
  (let [frm (get-frame state)]
    (if (get-in frm [:children who])
      state
      (update-frame state assoc-in [:children who]
                    (assoc (frame who (:variant frm)) :seq (next-seq (:children frm)))))))

(defn enter [state who] (update state :path conj who))
(defn leave [state] (update state :path pop))

(defn frame-context
  "The frame an error was found in, described as Python memo's error notes do."
  [state]
  (let [fr (get-frame state)
        chain (->> (range (count (:path state)) 0 -1)
                   (map #(get-frame state (subvec (:path state) 0 %)))
                   (map :name))
        modeled-by (apply str (map #(str ", as modeled by " %) (rest chain)))
        choices (map (fn [[[w i] _]] (if (= w 'self) (str i) (str w "." i)))
                     (:choices fr))]
    (str "This error was encountered in the frame of " (:name fr) modeled-by
         ". In that frame, " (:name fr) " is currently modeling the following "
         (count choices) " choices: " (str/join ", " choices) ".")))
