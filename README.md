
## Turing Patterns

One of the interests of Alan Turing (1912-1954) was pattern formation, for
example, how do leopards get their spots? He proposed a set of
reaction-diffusion equations, which can produce these patterns, depending
on the input parameters.

A simple implementation can be made by taking an initial random
distribution of "pigment", and applying two gaussian blurs of
different radii, subtracting them, applying a limiter, and use this to
update the distribution.

This repository contains a Python implentation, which can also take an
input image to work on.