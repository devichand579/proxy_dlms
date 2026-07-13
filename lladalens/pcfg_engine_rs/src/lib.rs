use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustc_hash::FxHashMap;
use std::collections::HashMap;

mod sample;

const EPS: f64 = 1e-12;

#[derive(Clone, Hash, PartialEq, Eq)]
struct CommitKey {
    l: usize,
    items: Vec<(usize, usize)>,
}

impl CommitKey {
    fn new(l: usize, committed: &HashMap<usize, usize>) -> Self {
        let mut v: Vec<_> = committed.iter().map(|(&a, &b)| (a, b)).collect();
        v.sort_unstable();
        CommitKey { l, items: v }
    }
}

struct Charts {
    inside: Vec<f64>,
    outside: Option<Vec<f64>>,
    z: f64,
    l: usize,
    n: usize,
}

struct EngineCore {
    v: usize,
    n: usize,
    start: usize,
    emit: Vec<f64>,
    msum: Vec<f64>,
    aidx: Vec<usize>,
    bidx: Vec<usize>,
    cidx: Vec<usize>,
    pr: Vec<f64>,
}

impl EngineCore {
    fn leaf(&self, clamp: &HashMap<usize, usize>, i: usize) -> Vec<f64> {
        let mut row = vec![0.0; self.n];
        if let Some(&t) = clamp.get(&i) {
            for a in 0..self.n {
                row[a] = self.emit[t * self.n + a];
            }
        } else {
            row.copy_from_slice(&self.msum);
        }
        row
    }

    fn fill_inside(&self, l: usize, clamp: &HashMap<usize, usize>) -> (Vec<f64>, f64) {
        let n = self.n;
        let mut ins = vec![0.0; l * l * n];
        let idx = |i: usize, j: usize| (i * l + j) * n;

        for i in 0..l {
            let row = self.leaf(clamp, i);
            ins[idx(i, i)..idx(i, i) + n].copy_from_slice(&row);
        }

        for span in 2..=l {
            for i in 0..=l - span {
                let j = i + span - 1;
                let mut cell = vec![0.0; n];
                for k in i..j {
                    for r in 0..self.pr.len() {
                        let a = self.aidx[r];
                        let b = self.bidx[r];
                        let c = self.cidx[r];
                        let p = self.pr[r];
                        let left = ins[idx(i, k) + b];
                        let right = ins[idx(k + 1, j) + c];
                        cell[a] += p * left * right;
                    }
                }
                ins[idx(i, j)..idx(i, j) + n].copy_from_slice(&cell);
            }
        }
        let z = ins[idx(0, l - 1) + self.start];
        (ins, z)
    }

    fn fill_outside(&self, l: usize, inside: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut out = vec![0.0; l * l * n];
        let idx = |i: usize, j: usize| (i * l + j) * n;
        out[idx(0, l - 1) + self.start] = 1.0;

        for width in (0..l).rev() {
            for i in 0..=l - width - 1 {
                let j = i + width;
                let mut cell = out[idx(i, j)..idx(i, j) + n].to_vec();
                for k in (j + 1)..l {
                    for r in 0..self.pr.len() {
                        let a = self.aidx[r];
                        let b = self.bidx[r];
                        let c = self.cidx[r];
                        let p = self.pr[r];
                        cell[b] += p * out[idx(i, k) + a] * inside[idx(j + 1, k) + c];
                    }
                }
                for k in 0..i {
                    for r in 0..self.pr.len() {
                        let a = self.aidx[r];
                        let b = self.bidx[r];
                        let c = self.cidx[r];
                        let p = self.pr[r];
                        cell[c] += p * inside[idx(k, i - 1) + b] * out[idx(k, j) + a];
                    }
                }
                out[idx(i, j)..idx(i, j) + n].copy_from_slice(&cell);
            }
        }
        out
    }

    fn recompute_inside_covering(&self, ins: &mut [f64], l: usize, clamp: &HashMap<usize, usize>, pos: usize) {
        let n = self.n;
        let idx = |i: usize, j: usize| (i * l + j) * n;
        for span in 1..=l {
            for i in 0..=l - span {
                let j = i + span - 1;
                if pos < i || pos > j {
                    continue;
                }
                if span == 1 {
                    let row = self.leaf(clamp, i);
                    ins[idx(i, j)..idx(i, j) + n].copy_from_slice(&row);
                } else {
                    let mut cell = vec![0.0; n];
                    for k in i..j {
                        for r in 0..self.pr.len() {
                            let a = self.aidx[r];
                            let b = self.bidx[r];
                            let c = self.cidx[r];
                            let p = self.pr[r];
                            cell[a] += p * ins[idx(i, k) + b] * ins[idx(k + 1, j) + c];
                        }
                    }
                    ins[idx(i, j)..idx(i, j) + n].copy_from_slice(&cell);
                }
            }
        }
    }

    fn recompute_outside_covering(
        &self,
        out: &mut [f64],
        inside: &[f64],
        l: usize,
        pos: usize,
    ) {
        let n = self.n;
        let idx = |i: usize, j: usize| (i * l + j) * n;
        for width in (0..l).rev() {
            for i in 0..=l - width - 1 {
                let j = i + width;
                if pos < i || pos > j {
                    continue;
                }
                let mut cell = vec![0.0; n];
                for k in (j + 1)..l {
                    for r in 0..self.pr.len() {
                        let a = self.aidx[r];
                        let b = self.bidx[r];
                        let c = self.cidx[r];
                        let p = self.pr[r];
                        cell[b] += p * out[idx(i, k) + a] * inside[idx(j + 1, k) + c];
                    }
                }
                for k in 0..i {
                    for r in 0..self.pr.len() {
                        let a = self.aidx[r];
                        let b = self.bidx[r];
                        let c = self.cidx[r];
                        let p = self.pr[r];
                        cell[c] += p * inside[idx(k, i - 1) + b] * out[idx(k, j) + a];
                    }
                }
                out[idx(i, j)..idx(i, j) + n].copy_from_slice(&cell);
            }
        }
    }

    fn marginal_at(&self, _inside: &[f64], outside: &[f64], l: usize, z: f64, pos: usize) -> Vec<f64> {
        let mut m = vec![0.0; self.v];
        if z <= 0.0 {
            return m;
        }
        let idx = |i: usize, j: usize| (i * l + j) * self.n;
        let beta = &outside[idx(pos, pos)..idx(pos, pos) + self.n];
        for t in 0..self.v {
            let mut s = 0.0;
            for a in 0..self.n {
                s += self.emit[t * self.n + a] * beta[a];
            }
            m[t] = s / z;
        }
        let sum: f64 = m.iter().sum();
        if sum > 0.0 {
            for x in &mut m {
                *x /= sum;
            }
        }
        m
    }

    fn entropy(p: &[f64]) -> f64 {
        p.iter()
            .filter(|&&x| x > 0.0)
            .map(|&x| -x * (x + EPS).ln())
            .sum()
    }

    fn charts_cached(&self, cache: &mut FxHashMap<CommitKey, Charts>, l: usize, committed: &HashMap<usize, usize>) -> Charts {
        let key = CommitKey::new(l, committed);
        if let Some(c) = cache.get(&key) {
            return c.clone();
        }
        let (inside, z) = self.fill_inside(l, committed);
        let outside = if z > 0.0 {
            Some(self.fill_outside(l, &inside))
        } else {
            None
        };
        let pack = Charts { inside, outside, z, l, n: self.n };
        cache.insert(key, pack.clone());
        pack
    }

    fn charts_extend(
        &self,
        cache: &mut FxHashMap<CommitKey, Charts>,
        l: usize,
        parent_committed: &HashMap<usize, usize>,
        pos: usize,
        token: usize,
    ) -> Charts {
        let mut child_committed = parent_committed.clone();
        child_committed.insert(pos, token);
        let key = CommitKey::new(l, &child_committed);
        if let Some(c) = cache.get(&key) {
            return c.clone();
        }

        let parent_key = CommitKey::new(l, parent_committed);
        let parent = match cache.get(&parent_key) {
            Some(c) => c.clone(),
            None => return self.charts_cached(cache, l, &child_committed),
        };

        let mut new_inside = parent.inside.clone();
        self.recompute_inside_covering(&mut new_inside, l, &child_committed, pos);
        let idx = |i: usize, j: usize| (i * l + j) * self.n;
        let z = new_inside[idx(0, l - 1) + self.start];
        let outside = if z <= 0.0 {
            None
        } else if let Some(parent_out) = parent.outside.as_ref() {
            let mut new_outside = parent_out.clone();
            self.recompute_outside_covering(&mut new_outside, &new_inside, l, pos);
            Some(new_outside)
        } else {
            Some(self.fill_outside(l, &new_inside))
        };
        let pack = Charts {
            inside: new_inside,
            outside,
            z,
            l,
            n: self.n,
        };
        cache.insert(key, pack.clone());
        pack
    }

    fn all_marginals_map(&self, l: usize, committed: &HashMap<usize, usize>, cache: &mut FxHashMap<CommitKey, Charts>) -> HashMap<usize, Vec<f64>> {
        let charts = self.charts_cached(cache, l, committed);
        let mut out = HashMap::new();
        if charts.z <= 0.0 {
            return out;
        }
        let outside = charts.outside.as_ref().unwrap();
        for p in 0..l {
            if let Some(&t) = committed.get(&p) {
                let mut m = vec![0.0; self.v];
                m[t] = 1.0;
                out.insert(p, m);
            } else {
                out.insert(p, self.marginal_at(&charts.inside, outside, l, charts.z, p));
            }
        }
        out
    }

    fn joint_entropy_chain(
        &self,
        l: usize,
        committed: &HashMap<usize, usize>,
        s: &[usize],
        cache: &mut FxHashMap<CommitKey, Charts>,
        ent_memo: &mut FxHashMap<(CommitKey, Vec<usize>), f64>,
    ) -> f64 {
        if s.is_empty() {
            return 0.0;
        }
        let memo_key = (CommitKey::new(l, committed), s.to_vec());
        if let Some(&h) = ent_memo.get(&memo_key) {
            return h;
        }

        let h = if s.len() == 1 {
            let charts = self.charts_cached(cache, l, committed);
            if charts.z <= 0.0 {
                0.0
            } else {
                let outside = charts.outside.as_ref().unwrap();
                let m = self.marginal_at(&charts.inside, outside, l, charts.z, s[0]);
                Self::entropy(&m)
            }
        } else {
            let pos0 = s[0];
            let charts = self.charts_cached(cache, l, committed);
            if charts.z <= 0.0 {
                0.0
            } else {
                let outside = charts.outside.as_ref().unwrap();
                let m0 = self.marginal_at(&charts.inside, outside, l, charts.z, pos0);
                let mut acc = Self::entropy(&m0);
                for (v, &pv) in m0.iter().enumerate() {
                    if pv <= 0.0 {
                        continue;
                    }
                    self.charts_extend(cache, l, committed, pos0, v);
                    let mut cl = committed.clone();
                    cl.insert(pos0, v);
                    acc += pv * self.joint_entropy_chain(l, &cl, &s[1..], cache, ent_memo);
                }
                acc
            }
        };
        ent_memo.insert(memo_key, h);
        h
    }
}

#[pyclass]
struct PyPCFGState {
    core: EngineCore,
    l: usize,
    committed: HashMap<usize, usize>,
    inside: Vec<f64>,
    outside: Option<Vec<f64>>,
    z: f64,
}

#[pymethods]
impl PyPCFGState {
    #[getter]
    fn l(&self) -> usize {
        self.l
    }

    fn committed_dict(&self) -> HashMap<usize, usize> {
        self.committed.clone()
    }

    fn commit(&mut self, pos: usize, token: usize) {
        self.committed.insert(pos, token);
        self.core
            .recompute_inside_covering(&mut self.inside, self.l, &self.committed, pos);
        let n = self.core.n;
        let root = (self.l - 1) * n + self.core.start;
        self.z = self.inside[root];
        if let Some(out) = self.outside.as_mut() {
            let inside = self.inside.clone();
            self.core
                .recompute_outside_covering(out, &inside, self.l, pos);
        }
    }

    fn all_marginals<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if self.z <= 0.0 {
            return Ok(PyDict::new_bound(py));
        }
        if self.outside.is_none() {
            self.outside = Some(self.core.fill_outside(self.l, &self.inside));
        }
        let outside = self.outside.as_ref().unwrap();
        let dict = PyDict::new_bound(py);
        for p in 0..self.l {
            let m = if let Some(&t) = self.committed.get(&p) {
                let mut v = vec![0.0; self.core.v];
                v[t] = 1.0;
                v
            } else {
                self.core
                    .marginal_at(&self.inside, outside, self.l, self.z, p)
            };
            dict.set_item(p, m)?;
        }
        Ok(dict)
    }

    fn marginals_batch<'py>(&mut self, py: Python<'py>, query_positions: Vec<usize>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        if self.z <= 0.0 {
            return Ok(dict);
        }
        if self.outside.is_none() {
            self.outside = Some(self.core.fill_outside(self.l, &self.inside));
        }
        let outside = self.outside.as_ref().unwrap();
        for p in query_positions {
            let m = if let Some(&t) = self.committed.get(&p) {
                let mut v = vec![0.0; self.core.v];
                v[t] = 1.0;
                v
            } else {
                self.core
                    .marginal_at(&self.inside, outside, self.l, self.z, p)
            };
            dict.set_item(p, m)?;
        }
        Ok(dict)
    }
}

impl PyPCFGState {
    fn n(&self) -> usize {
        self.core.n
    }
}

#[pyclass]
struct PyPCFGEngine {
    core: EngineCore,
    t_idx: HashMap<String, usize>,
    cache: FxHashMap<CommitKey, Charts>,
}

#[pymethods]
impl PyPCFGEngine {
    #[new]
    fn new(cnf: &Bound<'_, PyAny>) -> PyResult<Self> {
        let terminals: Vec<String> = cnf.getattr("terminals")?.extract()?;
        let start: String = cnf.getattr("start")?.extract()?;
        let nt: Vec<String> = cnf.getattr("NT")?.extract()?;
        let term_rules: HashMap<String, HashMap<String, f64>> = cnf.getattr("term_rules")?.extract()?;
        let bin_rules: Vec<(String, String, String, f64)> = cnf.getattr("bin_rules")?.extract()?;

        let v = terminals.len();
        let n = nt.len();
        let nt_idx: HashMap<String, usize> = nt.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();
        let t_idx: HashMap<String, usize> = terminals.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();
        let start = *nt_idx.get(&start).ok_or_else(|| PyValueError::new_err("bad start"))?;

        let mut emit = vec![0.0; v * n];
        let mut msum = vec![0.0; n];
        for (a, d) in term_rules {
            let ai = nt_idx[&a];
            for (t, p) in d {
                let ti = t_idx[&t];
                emit[ti * n + ai] += p;
                msum[ai] += p;
            }
        }

        let mut aidx = Vec::with_capacity(bin_rules.len());
        let mut bidx = Vec::with_capacity(bin_rules.len());
        let mut cidx = Vec::with_capacity(bin_rules.len());
        let mut pr = Vec::with_capacity(bin_rules.len());
        for (h, b, c, p) in bin_rules {
            aidx.push(nt_idx[&h]);
            bidx.push(nt_idx[&b]);
            cidx.push(nt_idx[&c]);
            pr.push(p);
        }

        Ok(PyPCFGEngine {
            core: EngineCore {
                v,
                n,
                start,
                emit,
                msum,
                aidx,
                bidx,
                cidx,
                pr,
            },
            t_idx,
            cache: FxHashMap::default(),
        })
    }

    #[getter]
    fn V(&self) -> usize {
        self.core.v
    }

    #[getter]
    fn n(&self) -> usize {
        self.core.n
    }

    #[getter]
    fn t_idx(&self) -> HashMap<String, usize> {
        self.t_idx.clone()
    }

    #[getter]
    fn cnf(&self) -> PyResult<Py<PyAny>> {
        Err(PyValueError::new_err("cnf not exposed from rust engine"))
    }

    fn clear_cache(&mut self) {
        self.cache.clear();
    }

    fn Z(&mut self, l: usize, committed: HashMap<usize, usize>) -> f64 {
        let (inside, z) = self.core.fill_inside(l, &committed);
        let _ = inside;
        z
    }

    fn all_marginals<'py>(&mut self, py: Python<'py>, l: usize, committed: HashMap<usize, usize>) -> PyResult<Bound<'py, PyDict>> {
        let map = self.core.all_marginals_map(l, &committed, &mut self.cache);
        let dict = PyDict::new_bound(py);
        for (p, m) in map {
            dict.set_item(p, m)?;
        }
        Ok(dict)
    }

    fn marginals_batch<'py>(
        &mut self,
        py: Python<'py>,
        l: usize,
        committed: HashMap<usize, usize>,
        query_positions: Vec<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let charts = self.core.charts_cached(&mut self.cache, l, &committed);
        let dict = PyDict::new_bound(py);
        if charts.z <= 0.0 {
            return Ok(dict);
        }
        let outside = charts.outside.as_ref().unwrap();
        for p in query_positions {
            let m = if let Some(&t) = committed.get(&p) {
                let mut v = vec![0.0; self.core.v];
                v[t] = 1.0;
                v
            } else {
                self.core
                    .marginal_at(&charts.inside, outside, l, charts.z, p)
            };
            dict.set_item(p, m)?;
        }
        Ok(dict)
    }

    fn Mt(&mut self, l: usize, committed: HashMap<usize, usize>, s: Vec<usize>, model: Vec<f64>) -> PyResult<f64> {
        let all = self.core.all_marginals_map(l, &committed, &mut self.cache);
        let mut tot = 0.0;
        for pos in s {
            if let Some(qm) = all.get(&pos) {
                let mut pm = model.clone();
                let sum: f64 = pm.iter().sum();
                if sum > 0.0 {
                    for x in &mut pm {
                        *x /= sum;
                    }
                }
                tot += qm
                    .iter()
                    .zip(pm.iter())
                    .filter(|(&p, _)| p > 0.0)
                    .map(|(&p, &q)| p * ((p + EPS).ln() - (q + EPS).ln()))
                    .sum::<f64>();
            }
        }
        Ok(tot)
    }

    fn Jt_entropy(&mut self, l: usize, committed: HashMap<usize, usize>, s: Vec<usize>) -> f64 {
        if s.len() < 2 {
            return 0.0;
        }
        let mut s = s;
        s.sort_unstable();
        self.cache.clear();
        let _ = self.core.charts_cached(&mut self.cache, l, &committed);
        let h_marg: f64 = s
            .iter()
            .map(|&p| {
                let charts = self.core.charts_cached(&mut self.cache, l, &committed);
                if charts.z <= 0.0 {
                    return 0.0;
                }
                let outside = charts.outside.as_ref().unwrap();
                let m = self.core.marginal_at(&charts.inside, outside, l, charts.z, p);
                EngineCore::entropy(&m)
            })
            .sum();
        let mut ent_memo = FxHashMap::default();
        let h_joint = self
            .core
            .joint_entropy_chain(l, &committed, &s, &mut self.cache, &mut ent_memo);
        (h_marg - h_joint).max(0.0)
    }

    fn make_state(&self, l: usize, committed: HashMap<usize, usize>) -> PyPCFGState {
        let (inside, z) = self.core.fill_inside(l, &committed);
        PyPCFGState {
            core: self.core.clone(),
            l,
            committed,
            inside,
            outside: None,
            z,
        }
    }
}

// EngineCore needs Clone for state
impl Clone for EngineCore {
    fn clone(&self) -> Self {
        EngineCore {
            v: self.v,
            n: self.n,
            start: self.start,
            emit: self.emit.clone(),
            msum: self.msum.clone(),
            aidx: self.aidx.clone(),
            bidx: self.bidx.clone(),
            cidx: self.cidx.clone(),
            pr: self.pr.clone(),
        }
    }
}

impl Clone for Charts {
    fn clone(&self) -> Self {
        Charts {
            inside: self.inside.clone(),
            outside: self.outside.clone(),
            z: self.z,
            l: self.l,
            n: self.n,
        }
    }
}

#[pymodule]
fn pcfg_engine_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPCFGEngine>()?;
    m.add_class::<PyPCFGState>()?;
    sample::register(m)?;
    Ok(())
}
