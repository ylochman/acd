import Pkg

add_packages = packages -> for pkg=packages Pkg.add(pkg) end

add_packages(["LinearAlgebra", "StatsBase"])
add_packages(["MAT"]) # matlab data processing (optional, used in demo)
