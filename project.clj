(defproject gradient-descent "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [incanter "1.5.7"]
                 [net.mikera/core.matrix "0.57.0"]
                 [org.clojure/data.csv "0.1.3"]]
  :target-path "target/%s"
  :profiles {:dev {:dependencies [[lein-light-nrepl "0.3.3"]
                                  [enlive "1.1.6"]
                                  [cheshire "5.8.0"]
                                  [criterium "0.4.4"]]}}
  :repl-options {:nrepl-middleware [lighttable.nrepl.handler/lighttable-ops]})
