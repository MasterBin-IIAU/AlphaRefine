import os
import glob
import traceback

import torch
from ltr.trainers import LTRTrainer
from ltr.admin import loading, multigpu


class ARTrainer(LTRTrainer):
    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.module_name, self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)

    def save_checkpoint(self, epoch, save_interval):
        """Saves a checkpoint of the network and other variables."""

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__
        state = {
            'epoch': self.epoch,
            'actor_type': actor_type,
            'net_type': net_type,
            'net': net.state_dict(),
            'net_info': getattr(net, 'info', None),
            'constructor': getattr(net, 'constructor', None),
            'optimizer': self.optimizer.state_dict(),
            'stats': self.stats,
            'settings': self.settings
        }

        directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # First save as a tmp file
        if (epoch % save_interval) == 0:
            ckpt_path = '{}/{}_ep{:04d}.pth.tar'.format(directory, net_type, self.epoch)
            torch.save(state, ckpt_path)

        tmp_file_path = '{}/latest.pth.tar'.format(directory)
        torch.save(state, tmp_file_path)

    def load_checkpoint(self, checkpoint=None, fields=None, ignore_fields=None, load_constructor=False):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__

        if checkpoint is None:
            ckpt_latest = '{}/{}/latest.pth.tar'.format(self._checkpoint_dir, self.settings.project_path, net_type)
            if os.path.exists(ckpt_latest):
                checkpoint_path = ckpt_latest

            else:
                # Load most recent checkpoint
                checkpoint_list = sorted(glob.glob('{}/{}/{}_ep*.pth.tar'.format(self._checkpoint_dir,
                                                                                 self.settings.project_path, net_type)))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    print('No matching checkpoint file found')
                    return

            print("latest checkpoint found ({})".format(checkpoint_path))

        elif isinstance(checkpoint, int):
            # Checkpoint is the epoch number
            checkpoint_path = '{}/{}/{}_ep{:04d}.pth.tar'.format(self._checkpoint_dir, self.settings.project_path,
                                                                 net_type, checkpoint)
        elif isinstance(checkpoint, str):
            # checkpoint is the path
            if os.path.isdir(checkpoint):
                checkpoint_list = sorted(glob.glob('{}/*_ep*.pth.tar'.format(checkpoint)))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise Exception('No checkpoint found')
            else:
                checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        # Load network
        checkpoint_dict = loading.torch_load_legacy(checkpoint_path)

        assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'

        if fields is None:
            fields = checkpoint_dict.keys()
        if ignore_fields is None:
            ignore_fields = ['settings']

            # Never load the scheduler. It exists in older checkpoints.
        ignore_fields.extend(['lr_scheduler', 'constructor', 'net_type', 'actor_type', 'net_info'])

        # Load all fields
        for key in fields:
            if key in ignore_fields:
                continue
            if key == 'net':
                net.load_state_dict(checkpoint_dict[key])
            elif key == 'optimizer':
                self.optimizer.load_state_dict(checkpoint_dict[key])
            else:
                setattr(self, key, checkpoint_dict[key])

        # Set the net info
        if load_constructor and 'constructor' in checkpoint_dict and checkpoint_dict['constructor'] is not None:
            net.constructor = checkpoint_dict['constructor']
        if 'net_info' in checkpoint_dict and checkpoint_dict['net_info'] is not None:
            net.info = checkpoint_dict['net_info']

        # Update the epoch in lr scheduler
        if 'epoch' in fields:
            self.lr_scheduler.last_epoch = self.epoch

        return True

    def train(self, max_epochs, load_latest=False, fail_safe=True, save_interval=5):
        """Do training for the given number of epochs.
        args:
            max_epochs - Max number of training epochs,
            load_latest - Bool indicating whether to resume from latest epoch.
            fail_safe - Bool indicating whether the training to automatically restart in case of any crashes.
        """

        epoch = -1
        num_tries = 10
        for i in range(num_tries):
            try:
                if load_latest:
                    self.load_checkpoint()

                for epoch in range(self.epoch + 1, max_epochs + 1):
                    self.epoch = epoch

                    self.train_epoch()

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    if self._checkpoint_dir:
                        if self.settings.local_rank == 0:
                            self.save_checkpoint(epoch, save_interval)
                print('Finished training!')
                return

            except:
                print('Training crashed at epoch {}'.format(epoch))
                if fail_safe:
                    self.epoch -= 1
                    load_latest = True
                    print('Traceback for the error!')
                    print(traceback.format_exc())
                    print('Restarting training from last epoch ...')
                else:
                    raise
