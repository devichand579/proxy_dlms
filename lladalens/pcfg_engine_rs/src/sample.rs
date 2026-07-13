use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Clone)]
struct Rule {
    rhs: Vec<u32>,
    cdf: f64,
}

struct SamplerCore {
    start: u32,
    terminal: Vec<bool>,
    names: Vec<String>,
    by_head: Vec<Vec<Rule>>,
    min_term: Vec<u32>,
    max_term: Vec<Vec<u32>>,
    max_depth: usize,
}

impl SamplerCore {
    fn from_python(grammar: &Bound<'_, PyAny>, max_depth: usize) -> PyResult<Self> {
        let rules_obj = grammar.getattr("rules")?;
        let start: String = grammar.getattr("start")?.extract()?;
        let terminals: std::collections::HashSet<String> = grammar.getattr("terminals")?.extract()?;

        let mut sym_idx: HashMap<String, u32> = HashMap::new();
        let mut names: Vec<String> = Vec::new();
        let mut terminal: Vec<bool> = Vec::new();

        let mut ensure = |s: &str, is_term: bool| -> u32 {
            if let Some(&id) = sym_idx.get(s) {
                return id;
            }
            let id = names.len() as u32;
            sym_idx.insert(s.to_string(), id);
            names.push(s.to_string());
            terminal.push(is_term);
            id
        };

        for t in &terminals {
            ensure(t, true);
        }

        let n_rules = rules_obj.len()?;
        let mut grouped: HashMap<u32, Vec<(Vec<u32>, f64)>> = HashMap::new();

        for i in 0..n_rules {
            let item = rules_obj.get_item(i)?;
            let head: String = item.get_item(0)?.extract()?;
            let rhs_obj = item.get_item(1)?;
            let prob: f64 = item.get_item(2)?.extract()?;
            let head_id = ensure(&head, terminals.contains(&head));
            let rhs_len = rhs_obj.len()?;
            let mut rhs = Vec::with_capacity(rhs_len);
            for j in 0..rhs_len {
                let sym: String = rhs_obj.get_item(j)?.extract()?;
                rhs.push(ensure(&sym, terminals.contains(&sym)));
            }
            grouped.entry(head_id).or_default().push((rhs, prob));
        }

        let start_id = *sym_idx
            .get(&start)
            .ok_or_else(|| PyValueError::new_err(format!("unknown start symbol {start}")))?;

        let n_syms = names.len();
        let mut by_head = vec![Vec::new(); n_syms];
        for (head, mut rs) in grouped {
            let mut cdf = 0.0;
            for (rhs, p) in rs.drain(..) {
                cdf += p;
                by_head[head as usize].push(Rule { rhs, cdf });
            }
            let last = by_head[head as usize].len().saturating_sub(1);
            if last < by_head[head as usize].len() {
                by_head[head as usize][last].cdf = 1.0;
            }
        }

        let min_term = compute_min_term(n_syms, &terminal, &by_head);
        let max_term = compute_max_term(n_syms, &terminal, &by_head, max_depth);

        Ok(SamplerCore {
            start: start_id,
            terminal,
            names,
            by_head,
            min_term,
            max_term,
            max_depth,
        })
    }

    fn sample_one(&self, min_len: usize, max_len: usize, rng: &mut StdRng) -> Option<Vec<String>> {
        const MAX_TRIES: usize = 200_000;
        for _ in 0..MAX_TRIES {
            match self.sample_once(max_len, min_len, rng) {
                Some(tokens) if (min_len..=max_len).contains(&tokens.len()) => {
                    return Some(tokens);
                }
                _ => {}
            }
        }
        None
    }

    fn sample_once(&self, max_len: usize, min_len: usize, rng: &mut StdRng) -> Option<Vec<String>> {
        let mut out: Vec<u32> = Vec::with_capacity(max_len);
        if self.expand(self.start, 0, max_len, min_len, &mut out, rng).is_err() {
            return None;
        }
        Some(
            out.iter()
                .map(|&id| self.names[id as usize].clone())
                .collect(),
        )
    }

    fn expand(
        &self,
        sym: u32,
        depth: usize,
        max_len: usize,
        min_len: usize,
        out: &mut Vec<u32>,
        rng: &mut StdRng,
    ) -> Result<(), ()> {
        if out.len() >= max_len {
            return Err(());
        }
        let si = sym as usize;
        if self.terminal[si] {
            out.push(sym);
            return Ok(());
        }
        if depth > self.max_depth {
            return Err(());
        }

        let cur_len = out.len();
        let rem = max_len - cur_len;
        if self.min_term[si] as usize > rem {
            return Err(());
        }
        let rem_depth = self.max_depth.saturating_sub(depth);
        if cur_len + (self.max_term[si][rem_depth] as usize) < min_len {
            return Err(());
        }

        let rules = &self.by_head[si];
        if rules.is_empty() {
            return Err(());
        }

        let x: f64 = rng.gen();
        let chosen = rules.iter().find(|r| x <= r.cdf).unwrap_or(&rules[rules.len() - 1]);
        for &child in &chosen.rhs {
            self.expand(child, depth + 1, max_len, min_len, out, rng)?;
        }
        Ok(())
    }
}

fn compute_min_term(n_syms: usize, terminal: &[bool], by_head: &[Vec<Rule>]) -> Vec<u32> {
    let mut min_t = vec![u32::MAX; n_syms];
    for (i, &is_term) in terminal.iter().enumerate() {
        if is_term {
            min_t[i] = 1;
        }
    }
    loop {
        let mut changed = false;
        for i in 0..n_syms {
            if terminal[i] {
                continue;
            }
            let mut best = u32::MAX;
            for rule in &by_head[i] {
                let mut sum = 0u32;
                let mut ok = true;
                for &s in &rule.rhs {
                    let v = min_t[s as usize];
                    if v == u32::MAX {
                        ok = false;
                        break;
                    }
                    sum = sum.saturating_add(v);
                }
                if ok {
                    best = best.min(sum);
                }
            }
            if best != u32::MAX && min_t[i] != best {
                min_t[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    min_t
}

fn compute_max_term(
    n_syms: usize,
    terminal: &[bool],
    by_head: &[Vec<Rule>],
    max_depth: usize,
) -> Vec<Vec<u32>> {
    let mut max_t = vec![vec![0u32; max_depth + 1]; n_syms];
    for d in 0..=max_depth {
        for i in 0..n_syms {
            if terminal[i] {
                max_t[i][d] = 1;
                continue;
            }
            if d == 0 {
                max_t[i][d] = 0;
                continue;
            }
            let mut best = 0u32;
            for rule in &by_head[i] {
                let mut sum = 0u32;
                for &s in &rule.rhs {
                    sum = sum.saturating_add(max_t[s as usize][d - 1]);
                }
                best = best.max(sum);
            }
            max_t[i][d] = best;
        }
    }
    max_t
}

#[pyclass]
pub struct PyPCFGSampler {
    core: SamplerCore,
    pool: rayon::ThreadPool,
}

#[pymethods]
impl PyPCFGSampler {
    #[new]
    fn new(grammar: &Bound<'_, PyAny>, max_depth: usize, threads: usize) -> PyResult<Self> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads.max(1))
            .build()
            .map_err(|e| PyValueError::new_err(format!("rayon pool: {e}")))?;
        Ok(PyPCFGSampler {
            core: SamplerCore::from_python(grammar, max_depth)?,
            pool,
        })
    }

    fn sample_batch(
        &self,
        n: usize,
        min_len: usize,
        max_len: usize,
        seed: u64,
    ) -> Vec<Vec<String>> {
        self.pool.install(|| {
            (0..n)
                .into_par_iter()
                .filter_map(|i| {
                    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(i as u64));
                    self.core.sample_one(min_len, max_len, &mut rng)
                })
                .collect()
        })
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPCFGSampler>()?;
    Ok(())
}
