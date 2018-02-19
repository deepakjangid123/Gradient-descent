(ns gradient-descent.multi-linear-regression
  (:require
    [clojure.core.matrix :as matrix]
    [clojure.data.csv :as csv]
    [clojure.java.io :as io]
    [incanter.charts :as charts]
    [incanter.core :refer [view]]))

(defn read-csv
  "Reads csv file and returns its data"
  []
  (with-open [in-file (io/reader "resources/multi-linear-regression.csv")]
    (doall
      (csv/read-csv in-file))))

;; Parameters needed for multi-linear-regression
(defn convert-to-numbers
  [m]
  (mapv #(mapv read-string %) m))
(def data (convert-to-numbers (read-csv)))
(def alpha 0.01)
(def iterations 1500)
(def xs (map pop data))
(def y (map last data))
(def number-of-examples (count xs))
(def initial-theta (matrix/transpose (matrix/matrix [0 0 0])))

(defn prefix-one
  [X]
  (mapv #(cons 1 %) X))

(def X (matrix/matrix xs))

(defn average
  [numbers]
  (/ (apply + numbers) (count numbers)))

(defn standarad-deviation
  [coll]
  (let [avg (average coll)
        squares (for [x coll]
                  (let [x-avg (- x avg)]
                    (* x-avg x-avg)))
        total (count coll)]
    (-> (/ (apply + squares)
           (dec total))
        (Math/sqrt))))

;; Now normalize the data
(defn normalise-row
  "Normalize the data so that gradient descent algo doesn't take too long to
  produce minima"
  [x]
  (let [mean (average x)
        std (standarad-deviation x)]
    (->> x
         (mapv #(- % mean))
         (mapv #(/ % std)))))

(defn normalise-features
  [X]
  (->> X
       (matrix/transpose)
       (mapv normalise-row)
       (matrix/transpose)))

(defn hypothesis
  "Returns hypothesis based on the given `theta` and `X` features"
  [X theta]
  (matrix/mmul X theta))

(defn mean-square-error
  "Returns mean-square-error for the predicted and actual values"
  [guess actual]
  (-> (matrix/sub guess actual)
      (matrix/square)
      (matrix/esum)))

(defn compute-cost
  "Computes cost fn J(theta)"
  [X y theta]
  (let [predicted (hypothesis X theta)]
    (/ (mean-square-error predicted y)
       (* 2 number-of-examples))))

(defn cost-derivative
  "Returns cost-derivative"
  [X y theta]
  (let [prediction (hypothesis X theta)]
    (-> prediction
        (matrix/sub y)
        (matrix/mmul X)
        (matrix/mul (/ alpha number-of-examples)))))

(defn gradient-descent
  "Gradient-descent algo entry point"
  [X y theta alpha number-of-iterations]
  (loop [remaining-iterations number-of-iterations
         theta theta]
    (if (zero? remaining-iterations)
      theta
      (recur (dec remaining-iterations)
             (matrix/sub theta (cost-derivative X y theta))))))

(defn costs
  "Keeps track of cost fn value at each step"
  [X y theta alpha number-of-iterations]
  (loop [remaining-iterations number-of-iterations
         theta theta
         costs []]
    (if (zero? remaining-iterations)
      costs
      (recur (dec remaining-iterations)
             (matrix/sub theta (cost-derivative X y theta))
             (conj costs (compute-cost X y theta))))))

(def Xa (-> X
            (normalise-features)
            (prefix-one)))

(def result (gradient-descent Xa y initial-theta alpha iterations))
(def cost-history (costs Xa y initial-theta alpha iterations))

(defn plot-cost-history
  "Plots cost-history at each and every step"
  [y]
  (let [plot (charts/scatter-plot (range 0 iterations) y
                                  :x-label "Iteration"
                                  :y-label "Cost")]
    (doto plot view)))

(plot-cost-history cost-history)

(defn plot-data
  "Plots data"
  [x y z]
  (let [plot (charts/scatter-plot x y
                                  :x-label "Population of City in 10,000s"
                                  :y-label "Profit in $10,000s")]
    (doto plot
      (charts/add-function #(+ (first result) (* (second result) %)) 0 25)
      view)))

