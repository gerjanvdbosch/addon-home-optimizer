# CLAUDE.md

Guidelines for coding agents in this repository.

## Core Principles

* Build **physically correct models** with a clear, simple, and maintainable
  architecture.
* Write **as little code as practically possible**, without compromising correctness,
  clarity, or architecture.
* Where possible, choose a simple and efficient solution over additional abstractions or
  complexity.
* Reuse existing code; avoid duplicate logic and unnecessary refactors. Check for
  existing functionality before adding new code.
* Understand the existing code and architecture before making changes.

## Physical System

### Heat Pump

* Minimize compressor starts; combine compatible demands within one compressor run where
  possible.
* Heating and DHW may be served sequentially within the same compressor run, but are not
  simultaneous.
* DHW also heats the space; perform DHW after cooling to avoid condensation.

### Boiler

* Stratified tank with top and bottom temperature sensors.
* Heating the boiler causes mixing, especially when heating starts.
* Heat is supplied to the boiler at the bottom.

## Physics

* Every state and parameter has a clear physical meaning and unit.
* All equations and terms are dimensionally consistent.
* Energy, mass, and other relevant balances must be physically correct.
* Every model term has a physical meaning; do not add terms or correction factors solely
  for a better fit.
* Parameter values and bounds must be physically realistic and justified.
* Explicitly state assumptions and simplifications.
* Do not use magic numbers. Model constants must have a clear physical meaning, unit,
  and physical justification.

## ODEs and Models

* Derive ODEs from physical balances: **inflow − outflow + production − consumption =
  accumulation**.
* Coupled transfers must be physically, dimensionally, and sign-consistent.
* Keep model definition, physical processes, balances, and numerical integration
  logically separated where this adds value.
* The discrete implementation must retain the same physical meaning as the continuous
  formulation.

## Validation

* Test new or modified models on independent data when available.
* Check not only the fit but also physical plausibility, conservation laws, and edge
  cases.
* A better fit is not evidence of a better physical model.
* Actually run relevant tests and checks.
* Prefer lightweight tests and validation.

## Code

* Use clear names and simple, efficient code.
* Avoid unnecessary functions, classes, abstractions, and configuration.
* Add abstractions only when they justify a clear responsibility or reuse.
* Comments should primarily describe **why**, physical assumptions, and non-trivial
  choices.
* Avoid redundant, trivial, or descriptive comments that merely repeat what the code
  does.
* When in doubt, choose the **simplest solution that is physically correct and
  architecturally sound**.
* Do not invent missing physics, parameters, or assumptions.
