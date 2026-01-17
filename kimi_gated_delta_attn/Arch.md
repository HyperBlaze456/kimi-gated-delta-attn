The architecture solely follows the definition present in [this file](references/modeling.py)

However, core Gated DeltaNet based linear attention functions are inside FLA project, which is super hard to decypher.
They are written in triton, hard to read with all those fused calculations and such.

Moreover, the Kimi Delta Attention uses specialized modules such as ShortConvolution and FusedRMSNormGated.

This architecture explanation would guide through what each module actually does, removing the ambiguity.