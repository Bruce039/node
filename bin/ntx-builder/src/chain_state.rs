use std::sync::{Arc, RwLock};

use miden_protocol::block::{BlockHeader, BlockNumber, SignedBlock, SignedBlockError};
use miden_protocol::crypto::merkle::mmr::PartialMmr;
use miden_protocol::transaction::PartialBlockchain;

// CHAIN STATE
// ================================================================================================

/// Contains information about the chain that is relevant to the [`NetworkTransactionBuilder`] and
/// all account actors managed by the [`Coordinator`].
///
/// The chain MMR stored here contains:
/// - The MMR peaks.
/// - Block headers and authentication paths for the last
///   [`NtxBuilderConfig::max_block_count`](crate::NtxBuilderConfig::max_block_count) blocks.
///
/// Authentication paths for older blocks are pruned because the NTX builder executes all notes as
/// "unauthenticated" (see [`InputNotes::from_unauthenticated_notes`]) and therefore does not need
/// to prove that input notes were created in specific past blocks.
#[derive(Debug, Clone)]
pub struct ChainState {
    /// The current tip of the chain.
    pub chain_tip_header: BlockHeader,
    /// A partial representation of the chain MMR.
    ///
    /// Contains block headers and authentication paths for the last
    /// [`NtxBuilderConfig::max_block_count`](crate::NtxBuilderConfig::max_block_count) blocks
    /// only, since all notes are executed as unauthenticated.
    pub chain_mmr: Arc<PartialBlockchain>,
}

impl ChainState {
    /// Constructs a new instance of [`ChainState`].
    pub(crate) fn new(chain_tip_header: BlockHeader, chain_mmr: PartialMmr) -> Self {
        let chain_mmr = PartialBlockchain::new(chain_mmr, [])
            .expect("partial blockchain should build from partial mmr");
        Self {
            chain_tip_header,
            chain_mmr: Arc::new(chain_mmr),
        }
    }

    /// Consumes the chain state and returns the chain tip header and the partial blockchain as a
    /// tuple.
    pub fn into_parts(self) -> (BlockHeader, Arc<PartialBlockchain>) {
        (self.chain_tip_header, self.chain_mmr)
    }

    /// Returns a clone of the current partial chain MMR.
    pub(crate) fn current_mmr(&self) -> PartialMmr {
        self.chain_mmr.mmr().clone()
    }

    /// Verifies the block against the current tip before updating the chain MMR.
    pub(crate) fn update_chain_tip(
        &mut self,
        block: &SignedBlock,
        max_block_count: usize,
    ) -> Result<(), SignedBlockError> {
        block.validate(Some(&self.chain_tip_header))?;

        // Update MMR which lags by one block.
        let mmr_tip = self.chain_tip_header.clone();
        Arc::make_mut(&mut self.chain_mmr).add_block(&mmr_tip, true);

        // Set the new tip.
        self.chain_tip_header = block.header().clone();

        // Keep MMR pruned.
        let pruned_block_height =
            (self.chain_mmr.chain_length().as_usize().saturating_sub(max_block_count)) as u32;
        Arc::make_mut(&mut self.chain_mmr).prune_to(..pruned_block_height.into());
        Ok(())
    }
}

/// A thread-safe wrapper around [`ChainState`] that can be shared across multiple actors.
///
/// The API guarantees that the lock cannot be held across await points.
pub struct SharedChainState(RwLock<ChainState>);

impl SharedChainState {
    pub fn new(chain_tip_header: BlockHeader, chain_mmr: PartialMmr) -> Self {
        Self(RwLock::new(ChainState::new(chain_tip_header, chain_mmr)))
    }

    pub(crate) fn chain_tip_block_number(&self) -> BlockNumber {
        self.0.read().expect("chain state lock poisoned").chain_tip_header.block_num()
    }

    /// Returns a clone of the current partial chain MMR. Cheap enough for per-block persistence
    /// since the MMR is bounded by `max_block_count` headers.
    pub(crate) fn current_mmr(&self) -> PartialMmr {
        self.0.read().expect("chain state lock poisoned").current_mmr()
    }

    pub(crate) fn update_chain_tip(
        &self,
        block: &SignedBlock,
        max_block_count: usize,
    ) -> Result<(), SignedBlockError> {
        self.0
            .write()
            .expect("chain state lock poisoned")
            .update_chain_tip(block, max_block_count)
    }

    pub(crate) fn get_cloned(&self) -> ChainState {
        self.0.read().expect("chain state lock poisoned").clone()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use miden_node_store::GenesisState;
    use miden_node_utils::fee::{test_fee_params, test_protocol_config};
    use miden_protocol::block::{BlockInputs, BlockSignatures, ProposedBlock, ValidatorConfig};
    use miden_protocol::crypto::dsa::ecdsa_k256_keccak::SigningKey;

    use super::*;

    #[test]
    fn rejected_blocks_do_not_advance_chain_state() {
        let signer = SigningKey::new();
        let attacker = SigningKey::new();
        let genesis = GenesisState::new(
            vec![],
            test_fee_params(),
            0,
            ValidatorConfig::new(vec![signer.public_key()], 1).unwrap(),
            test_protocol_config(),
        )
        .into_block()
        .unwrap();
        let parent = genesis.inner().header();
        let chain = SharedChainState::new(parent.clone(), PartialMmr::default());
        let initial_mmr = chain.current_mmr();
        let inputs = BlockInputs::new(
            parent.clone(),
            PartialBlockchain::default(),
            BTreeMap::new(),
            BTreeMap::new(),
            BTreeMap::new(),
        );
        let (header, body) = ProposedBlock::new_at(inputs, vec![], parent.timestamp() + 1)
            .unwrap()
            .with_next_validator_config(
                ValidatorConfig::new(vec![attacker.public_key()], 1).unwrap(),
            )
            .into_header_and_body()
            .unwrap();

        for signatures in [
            vec![],
            vec![attacker.sign(header.commitment())],
            vec![signer.sign(parent.commitment())],
        ] {
            let invalid = SignedBlock::new_unchecked(
                header.clone(),
                body.clone(),
                BlockSignatures::new(signatures).unwrap(),
            );
            assert!(chain.update_chain_tip(&invalid, 4).is_err());
            assert_eq!(&chain.get_cloned().chain_tip_header, parent);
            assert_eq!(chain.current_mmr(), initial_mmr);
        }

        let signatures = BlockSignatures::new(vec![signer.sign(header.commitment())]).unwrap();
        let valid = SignedBlock::new_unchecked(header, body, signatures);
        chain.update_chain_tip(&valid, 4).unwrap();
        assert_eq!(&chain.get_cloned().chain_tip_header, valid.header());
        let advanced_mmr = chain.current_mmr();
        assert_ne!(advanced_mmr, initial_mmr);

        assert!(chain.update_chain_tip(&valid, 4).is_err());
        assert_eq!(&chain.get_cloned().chain_tip_header, valid.header());
        assert_eq!(chain.current_mmr(), advanced_mmr);
    }
}
