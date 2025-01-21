use crate::op::HasProcessTrace;
use luminal::prelude::*;

pub trait LuminairGraph {
    fn gen_trace(&mut self);
}

impl LuminairGraph for Graph {
    fn gen_trace(&mut self) {
        if self.linearized_graph.is_none() {
            self.toposort();
        }
        let mut consumers = self.consumers_map.as_ref().unwrap().clone();
        let mut dim_stack = Vec::new();

        for (node, src_ids) in self.linearized_graph.as_ref().unwrap() {
            if self.tensors.contains_key(&(*node, 0)) {
                continue;
            }

            let mut srcs =
                get_source_tensors(&self.no_delete, &mut self.tensors, src_ids, &consumers);

            // Substitute in the dyn dims
            for (_, st) in srcs.iter_mut() {
                st.resolve_global_dyn_dims_stack(&self.dyn_map, &mut dim_stack);
            }

            let node_op = &mut *self.graph.node_weight_mut(*node).unwrap();

            let tensors = if node_op.has_process_trace() {
                println!("Found operator with process_trace: {:?}", node_op);
                let (tensors, _trace) = node_op.call_process_trace(srcs).unwrap();
                tensors
            } else {
                println!("Using regular process: {:?}", node_op);
                node_op.process(srcs)
            };

            for (i, tensor) in tensors.into_iter().enumerate() {
                self.tensors.insert((*node, i as u8), tensor);
            }

            // Bookkeep remaining consumers
            for (id, ind, _) in src_ids {
                *consumers.get_mut(&(*id, *ind)).unwrap() -= 1;
            }
        }
        self.reset();
    }
}
