use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard, PoisonError};

use super::protocol::{Frame, Refusal, Spec};
use super::server::{CanvasId, Sink};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanvasView {
    pub id: CanvasId,
    pub spec: Spec,
    pub presented: u64,
    pub last: Option<Frame>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Totals {
    pub opened: u64,
    pub presented: u64,
    pub closed: u64,
}

#[derive(Default)]
struct State {
    live: BTreeMap<CanvasId, CanvasView>,
    totals: Totals,
}

#[derive(Default)]
pub struct HeadlessCanvases {
    state: Mutex<State>,
}

impl HeadlessCanvases {
    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    #[must_use]
    pub fn live(&self) -> Vec<CanvasView> {
        self.state().live.values().cloned().collect()
    }

    #[must_use]
    pub fn titled(&self, title: &str) -> Option<CanvasView> {
        self.state()
            .live
            .values()
            .find(|c| c.spec.title == title)
            .cloned()
    }

    #[must_use]
    pub fn totals(&self) -> Totals {
        self.state().totals
    }
}

impl Sink for HeadlessCanvases {
    fn open(&self, id: CanvasId, spec: &Spec) -> Result<(), Refusal> {
        let mut state = self.state();
        state.totals.opened += 1;
        state.live.insert(
            id,
            CanvasView {
                id,
                spec: spec.clone(),
                presented: 0,
                last: None,
            },
        );
        Ok(())
    }

    fn present(&self, id: CanvasId, frame: Frame) -> Result<(), Refusal> {
        let mut state = self.state();
        let canvas = state
            .live
            .get_mut(&id)
            .ok_or_else(|| Refusal::Unavailable(format!("canvas {id} is not held")))?;
        canvas.presented += 1;
        canvas.last = Some(frame);
        state.totals.presented += 1;
        Ok(())
    }

    fn close(&self, id: CanvasId) {
        let mut state = self.state();
        if state.live.remove(&id).is_some() {
            state.totals.closed += 1;
        }
    }
}
